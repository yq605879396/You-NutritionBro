import torch
import numpy as np
import pickle, os, json, sys, random
from tqdm import tqdm

# import from other module
from args import get_parser
from torchvision import transforms
from build_vocab import Vocabulary
from data_loader import get_loader
from model import get_model, mask_from_eos
from helper import label2onehot, save_model, count_parameters, set_lr, make_dir, update_error_types, compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'


def main(args):

    where_to_save = os.path.join(args.save_dir, args.project_name, args.model_name)
    checkpoints_dir = os.path.join(where_to_save, 'checkpoints')
    logs_dir = os.path.join(where_to_save, 'logs')

    if not args.log_term:
        sys.stdout = open(os.path.join(logs_dir, 'eval.log'), 'w')
        sys.stderr = open(os.path.join(logs_dir, 'eval.err'), 'w')

    transforms_list = []
    transforms_list.append(transforms.Resize((args.crop_size)))
    transforms_list.append(transforms.CenterCrop(args.crop_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))
    #Image preprocessing
    transform = transforms.Compose(transforms_list)

    #Data loader
    data_loader, dataset = get_loader(args.data_dir,'test',
                                      args.maxnumlabels,
                                      batch_size = args.batch_size,
                                      transform = transform,
                                      shuffle=False, num_workers=args.num_workers,
                                                          drop_last=False,
                                                          max_num_samples=-1)
    
    ingr_vocab_size = dataset.get_ingrs_vocab_size()

    args.numgens = 1

    #Build the model
    model = get_model(args, ingr_vocab_size)
    model_path = os.path.join(args.save_dir, args.project_name, args.model_name, 'checkpoints', 'modelbest.ckpt')
    model.load_state_dict(torch.load(model_path, map_location=map_loc))

    model.eval()
    model = model.to(device)
    error_types = {'tp_i': 0, 'fp_i': 0, 'fn_i': 0, 'tn_i': 0, 'tp_all': 0, 'fp_all': 0, 'fn_all': 0}

    for i, (img_inputs, ingr_gt, img_id, path) in tqdm(enumerate(data_loader)):

        ingr_gt = ingr_gt.to(device)
        img_inputs = img_inputs.to(device)

        for gens in range(args.numgens):
            with torch.no_grad():

                outputs = model.sample(img_inputs)

                fake_ingrs = outputs['ingr_ids']
                pred_one_hot = label2onehot(fake_ingrs, ingr_vocab_size - 1)
                target_one_hot = label2onehot(ingr_gt, ingr_vocab_size - 1)

                update_error_types(error_types, pred_one_hot, target_one_hot)
                    
    ret_metrics = {'accuracy': [], 'f1': []}
    compute_metrics(ret_metrics, error_types, ['accuracy', 'f1'],
                        eps=1e-10,
                        weights=None)

    for k, v in ret_metrics.items():
        print(k, np.mean(v))


if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)
