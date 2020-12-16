
# some basic library
import numpy as np
import os, random, pickle, sys, json, time

# other modules
from nutribro_model.model import get_model, mask_from_eos
from suply.args import get_parser
from suply.data_loader import get_loader
from suply.build_vocab import Vocabulary
from suply.hepler import label2onehotm, draw_result, save_model, count_paramerters, compute_metrics, update_error_types, make_dir

# for model
from torchvision import transforms
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.backends.cudnn as cudnn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

# set learning rate
def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr']*decay_factor

def main(args):

    #Create model directory
    where_to_save = os.path.join(args.save_dir, args.project_name, args.model_name)
    checkpoints_dir = os.path.join(where_to_save, 'checkpoints')
    
    logs_dir = os.path.join(where_to_save, 'logs')

    tb_logs = os.path.join(args.save_dir, args.project_name, 'tb_logs', args.model_name)
    make_dir(where_to_save)
    make_dir(logs_dir)
    make_dir(checkpoints_dir)
    make_dir(tb_logs)

    if not args.log_term:
        print ("Training logs will be saved to:", os.path.join(logs_dir, 'train.log'))
        sys.stdout = open(os.path.join(logs_dir, 'train.log'), 'w')
        sys.stderr = open(os.path.join(logs_dir, 'train.err'), 'w')
        
    print(args)
    pickle.dump(args, open(os.path.join(checkpoints_dir, 'args.pkl'), 'wb'))

    #Patience
    patience = 0

    #Data loader
    data_loaders = {}
    datasets = {}

    data_dir = args.data_dir
    for stage in ['train', 'val']:

        transforms_list = [transforms.Resize((args.image_size))]

        if stage == 'train':
            #Image preprocessing, normalization for the pretrained resnet
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
            transforms_list.append(transforms.RandomCrop(args.crop_size))

        else:
            transforms_list.append(transforms.CenterCrop(args.crop_size))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)))

        transform = transforms.Compose(transforms_list)
        max_num_samples = max(args.max_eval, args.batch_size) if stage == 'val' else -1
        
        data_loaders[stage], datasets[stage] = get_loader(data_dir, stage,
                                                          args.maxnumlabels,
                                                          transform, args.batch_size,
                                                          shuffle=stage == 'train',
                                                          num_workers=args.num_workers,
                                                          drop_last=True,
                                                          max_num_samples=max_num_samples)

    ingr_vocab_size = datasets[stage].get_ingrs_vocab_size()

    #Build the model
    model = get_model(args, ingr_vocab_size)
    keep_cnn_gradients = False

    decay_factor = 1.0

    params = list(model.ingredient_decoder.parameters())
    params_cnn = list(model.image_encoder.resnet.parameters())


    # Optimizing CNN
    if params_cnn is not None and args.finetune_after == 0:
        optimizer = torch.optim.Adam([{'params': params}, {'params': params_cnn,
                                                           'lr': args.learning_rate*args.scale_learning_rate_cnn}],
                                     lr=args.learning_rate, weight_decay=args.weight_decay)
        keep_cnn_gradients = True
        print ("Fine tuning resnet")
    else:
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)


    if device != 'cpu' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    cudnn.benchmark = True

    if args.es_metric == 'loss':
        es_best = 10000 
    else:
        es_best = 0

    # Train the model
    train_loss = {'loss':[], 'ingrt':[]}
    val_loss = {'loss':[], 'ingrt':[]}

    start = args.current_epoch
    for epoch in range(start, args.num_epochs):
        print("The " + str(epoch)+"th epoch")

        args.current_epoch = epoch

        if args.decay_lr:
            frac = epoch // args.lr_decay_every
            decay_factor = args.lr_decay_rate ** frac
            new_lr = args.learning_rate*decay_factor
            set_lr(optimizer, decay_factor)

        if args.finetune_after != -1 and args.finetune_after < epoch \
                and not keep_cnn_gradients and params_cnn is not None:

            print("Starting to fine tune CNN")
            #Start with learning rates given
            optimizer = torch.optim.Adam([{'params': params},
                                          {'params': params_cnn,
                                           'lr': decay_factor*args.learning_rate*args.scale_learning_rate_cnn}],
                                         lr=decay_factor*args.learning_rate)
            keep_cnn_gradients = True

        for stage in ['train', 'val']:

            if stage == 'train':
                model.train()
            else:
                model.eval()
            total_step = len(data_loaders[stage])
            loader = iter(data_loaders[stage])

            total_loss_dict = {'ingr_loss': [],
                               'loss': []}

            error_types = {'tp_i': 0, 'fp_i': 0, 'fn_i': 0, 'tn_i': 0,
                           'tp_all': 0, 'fp_all': 0, 'fn_all': 0}

            torch.cuda.synchronize()
            start = time.time()

            for i in range(total_step):

                img_inputs, ingr_gt, img_id, path = loader.next()

                ingr_gt = ingr_gt.to(device)
                img_inputs = img_inputs.to(device)
                loss_dict = {}

                if stage == 'val':
                    with torch.no_grad():
                        losses = model(img_inputs, ingr_gt)

                        outputs = model(img_inputs, ingr_gt, sample=True)
                        ingr_ids_greedy = outputs['ingr_ids']
                        mask = mask_from_eos(ingr_ids_greedy, eos_value=0, mult_before=False)
                        ingr_ids_greedy[mask == 0] = ingr_vocab_size-1
                        pred_one_hot = label2onehot(ingr_ids_greedy, ingr_vocab_size-1)
                        target_one_hot = label2onehot(ingr_gt, ingr_vocab_size-1)
                        update_error_types(error_types, pred_one_hot, target_one_hot)
                        del outputs, pred_one_hot, target_one_hot, iou_sample
                else:
                    losses = model(img_inputs, ingr_gt,
                                   keep_cnn_gradients=keep_cnn_gradients)


                ingr_loss = losses['ingr_loss']
                ingr_loss = ingr_loss.mean()
                loss_dict['ingr_loss'] = ingr_loss.item()

                loss = 1000 * ingr_loss

                loss_dict['loss'] = loss.item()

                for key in loss_dict.keys():
                    total_loss_dict[key].append(loss_dict[key])

                if stage == 'train':
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                del loss, losses, img_inputs


            if stage == 'train':
                train_loss['loss'].append(np.mean(total_loss_dict['loss']))
                train_loss['ingrt'].append(np.mean(total_loss_dict['ingr_loss']))
            else:
                val_loss['loss'].append(np.mean(total_loss_dict['loss']))
                val_loss['ingrt'].append(np.mean(total_loss_dict['ingr_loss']))
                

            if stage == 'val':
                ret_metrics = {'accuracy': [], 'f1': [], 'jaccard': [], 'dice': []}
                compute_metrics(ret_metrics, error_types,
                                ['accuracy', 'f1', 'jaccard', 'dice'], eps=1e-10,
                                weights=None)

        es_value = np.mean(total_loss_dict[args.es_metric])

        #Save models
        save_model(model, optimizer, checkpoints_dir, suff='')
        if args.es_metric == 'loss' and es_value < es_best:
            es_best = es_value
            save_model(model, optimizer, checkpoints_dir, suff='best')
            pickle.dump(args, open(os.path.join(checkpoints_dir, 'args.pkl'), 'wb'))
            patience = 0
            print('Saved checkpoint.')
        else:
            patience += 1

        if patience > args.patience:
            break

    return train_loss, val_loss


if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    train_loss, val_loss = main(args)
    draw_result(args.num_epochs, train_loss, val_loss)


    