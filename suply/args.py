import argparse
import os

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='./result',
                        help='path where checkpoints will be saved')

    parser.add_argument('--project_name', type=str, default='find_ingredient',
                        help='name of the directory where models will be saved within save_dir')

    parser.add_argument('--model_name', type=str, default='ingredient_QIXI',
                        help='save_dir/project_name/model_name will be the path where logs and checkpoints are stored')

    parser.add_argument('--transfer_from', type=str, default='',
                        help='specify model name to transfer from')

    parser.add_argument('--suff', type=str, default='',
                        help='the id of the dictionary to load for training')

    parser.add_argument('--image_model', type=str, default='resnet50', choices=['resnet18', 'resnet50'])

    parser.add_argument('--data_dir', type=str, default='.',
                        help='directory where dataset is extracted')

    parser.add_argument('--aux_data_dir', type=str, default='../data',
                        help='path to other necessary data files (eg. vocabularies)')

    parser.add_argument('--crop_size', type=int, default=360, help='size for randomly or center cropping images')

    parser.add_argument('--image_size', type=int, default=384, help='size to rescale images')

    parser.add_argument('--log_step', type=int , default=10, help='step size for printing log info')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='base learning rate')

    parser.add_argument('--scale_learning_rate_cnn', type=float, default=1.0,
                        help='lr multiplier for cnn weights')

    parser.add_argument('--lr_decay_rate', type=float, default=0.99,
                        help='learning rate decay factor')

    parser.add_argument('--lr_decay_every', type=int, default=1,
                        help='frequency of learning rate decay (default is every epoch)')

    parser.add_argument('--weight_decay', type=float, default=0.)

    parser.add_argument('--embed_size', type=int, default=512,
                        help='hidden size for all projections')

    parser.add_argument('--n_att_ingrs', type=int, default=4,
                        help='number of attention heads in the ingredient decoder')

    parser.add_argument('--transf_layers', type=int, default=32,
                        help='number of transformer layers in the instruction decoder')

    parser.add_argument('--transf_layers_ingrs', type=int, default=4,
                        help='number of transformer layers in the ingredient decoder')

    parser.add_argument('--num_epochs', type=int, default=3,
                        help='maximum number of epochs')

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--dropout_encoder', type=float, default=0.3,
                        help='dropout ratio for the image and ingredient encoders')

    parser.add_argument('--dropout_decoder_i', type=float, default=0.3,
                        help='dropout ratio in the ingredient decoder')

    parser.add_argument('--finetune_after', type=int, default=-1,
                        help='epoch to start training cnn. -1 is never, 0 is from the beginning')

    parser.add_argument('--loss_weight', nargs='+', type=float, default=[0.0, 1000.0, 1.0, 1.0],
                        help='training loss weights. 1) instruction, 2) ingredient, 3) eos 4) cardinality')

    parser.add_argument('--max_eval', type=int, default=4096,
                        help='number of validation samples to evaluate during training')

    parser.add_argument('--patience', type=int, default=50,
                        help='maximum number of epochs to allow before early stopping')

    parser.add_argument('--maxnumlabels', type=int, default=30,
                        help='maximum number of ingredients per sample')

    parser.add_argument('--es_metric', type=str, default='iou_sample', choices=['loss', 'iou_sample'],
                        help='early stopping metric to track')

    parser.add_argument('--log_term', dest='log_term', action='store_true',
                        help='if used, shows training log in stdout instead of saving it to a file.')
    parser.set_defaults(log_term=False)

    parser.add_argument('--nodecay_lr', dest='decay_lr', action='store_false',
                        help='disables learning rate decay')
    parser.set_defaults(decay_lr=True)

    args = parser.parse_args()

    return args
