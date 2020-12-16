
import torch
import torch.nn as nn
import random
import numpy as np

from nutribro_model.transformer import Transformer
from nutribro_model.multihead_attention import MultiheadAttention
from suply.helper import label2onehot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mask_from_eos(ids, eos_value, mult_before=True):
    mask = torch.ones(ids.size()).to(device).byte()
    mask_aux = torch.ones(ids.size(0)).to(device).byte()

    #Search eos in ingredient prediction
    for idx in range(ids.size(1)):
        if idx == 0:
            continue
        if mult_before:
            mask[:, idx] = mask[:, idx] * mask_aux
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
        else:
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
            mask[:, idx] = mask[:, idx] * mask_aux
    return mask

# get whole modul = image encoder + ingredient decoder
def get_model(args, ingr_vocab_size):

    #Build image model
    encoder_image = EncoderCNN(args.embed_size, args.dropout_encoder, args.image_model)


    ingr_decoder = Transformer(args.embed_size, ingr_vocab_size, dropout=args.dropout_decoder_i,
                                      seq_length=args.maxnumlabels,
                                      attention_nheads=args.n_att_ingrs,
                                      num_layers=args.transf_layers_ingrs,
                                      scale_embed_grad=False)

    label_loss = nn.BCELoss(reduce=False)
    eos_loss = nn.BCELoss(reduce=False)

    model = IngredientsModel(ingr_decoder, encoder_image,
                                crit=criterion, crit_ingr=label_loss, crit_eos=eos_loss,
                                pad_value=ingr_vocab_size-1,label_smoothing=args.label_smoothing_ingr)
                                

    return model

# define encoder
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout=0.5, image_model='resnet101', pretrained=True):
        super(EncoderCNN, self).__init__()
        resnet = globals()[image_model](pretrained=pretrained)
        # Delete the last fc layer
        modules = list(resnet.children())[:-2] 
        self.resnet = nn.Sequential(*modules)

        self.linear = nn.Sequential(nn.Conv2d(resnet.fc.in_features, embed_size, kernel_size=1, padding=0),
                                    nn.Dropout2d(dropout))

    def forward(self, images, keep_cnn_gradients=False):

        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            with torch.no_grad():
                raw_conv_feats = self.resnet(images)
                
        features = self.linear(raw_conv_feats)
        features = features.view(features.size(0), features.size(1), -1)

        return features


class IngredientsModel(nn.Module):
    def __init__(self, ingr_decoder, image_encoder,
                 crit=None, crit_ingr=None, crit_eos=None,
                 pad_value=0, label_smoothing=0.0):

        super(IngredientsModel, self).__init__()

        self.ingredient_encoder = ingredient_encoder
        self.image_encoder = image_encoder
        self.ingredient_decoder = ingr_decoder
        self.crit_ingr = crit_ingr
        self.pad_value = pad_value
        self.crit_eos = crit_eos
        self.label_smoothing = label_smoothing


    def forward(self, img_inputs, target_ingrs,
                sample=False, keep_cnn_gradients=False):

        if sample:
            return self.sample(img_inputs, greedy=True)

        #Get image feature
        img_features = self.image_encoder(img_inputs, keep_cnn_gradients)

        losses = {}

        #Prepare groud truth ingredients
        target_one_hot = label2onehot(target_ingrs, self.pad_value)
        target_one_hot_smooth = label2onehot(target_ingrs, self.pad_value)

        target_one_hot_smooth[target_one_hot_smooth == 1] = (1-self.label_smoothing)
        target_one_hot_smooth[target_one_hot_smooth == 0] = self.label_smoothing / target_one_hot_smooth.size(-1)

        #Predict ingredients
        ingr_ids, ingr_logits = self.ingredient_decoder.sample(temperature=1.0, img_features=img_features,
                                                               first_token_value=0)

        ingr_logits = torch.nn.functional.softmax(ingr_logits, dim=-1)

        #Find idxs for eos ingredient
        #Assign eos probability to be in the first position of the softmax
        eos = ingr_logits[:, :, 0]
        target_eos = ((target_ingrs == 0) ^ (target_ingrs == self.pad_value))

        eos_pos = (target_ingrs == 0)
        eos_head = ((target_ingrs != self.pad_value) & (target_ingrs != 0))

        #Pooling the selected transformer steps
        mask_perminv = mask_from_eos(target_ingrs, eos_value=0, mult_before=False)
        ingr_probs = ingr_logits * mask_perminv.float().unsqueeze(-1)

        ingr_probs, _ = torch.max(ingr_probs, dim=1)

        #Ignore predicted ingredients after eos in ground truth
        ingr_ids[mask_perminv == 0] = self.pad_value

        ingr_loss = self.crit_ingr(ingr_probs, target_one_hot_smooth)
        ingr_loss = torch.mean(ingr_loss, dim=-1)

        losses['ingr_loss'] = ingr_loss

        return losses

    def sample(self, img_inputs, greedy=True, temperature=1.0, beam=-1, true_ingrs=None):

        outputs = dict()

        img_features = self.image_encoder(img_inputs)

        ingr_ids, ingr_probs = self.ingredient_decoder.sample(temperature=temperature,
                                                              img_features=img_features, first_token_value=0)

        #Mask ingredients after finding eos
        sample_mask = mask_from_eos(ingr_ids, eos_value=0, mult_before=False)
        ingr_ids[sample_mask == 0] = self.pad_value

        outputs['ingr_ids'] = ingr_ids
        outputs['ingr_probs'] = ingr_probs.data

        return outputs