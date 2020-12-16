import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nutribro_model.multihead_attention import MultiheadAttention
import numpy as np
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def LayerNorm(embed_d):
    m = nn.LayerNorm(embed_d)
    return m


def Linear(in_d, out_d, bias=True):
    m = nn.Linear(in_d, out_d, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.0)
    return m

class DecoderLayer(nn.Module):
    
    #Decoder layer block
    def __init__(self, embed_d, n_att, dropout=0.5):
        super().__init__()

        self.embed_d = embed_d
        self.dropout = dropout
        num_layer_norm = 3

        self.self_attention_net = MultiheadAttention(
            self.embed_d, n_att,
            dropout=dropout,
        )

        self.cond_att = MultiheadAttention(
            self.embed_d, n_att,
            dropout=dropout,
        )

        self.fc1 = Linear(self.embed_d, self.embed_d)
        self.fc2 = Linear(self.embed_d, self.embed_d)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_d) for i in range(num_layer_norm)])
        self.last_ln = LayerNorm(self.embed_d)
  

    def forward(self, x, incremental_state, img_features):

        residual_x = x

        #Norm Layer 1
        x = self.layer_norms[0](x)

        #Attention Layer 1
        x, _ = self.self_attention_net(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
            need_weights=False,
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual_x + x

        residual_x = x

        # Norm Layer 2
        x = self.layer_norms[1](x)
        
        # Attention Layer 2
        x, _ = self.cond_att(query=x,
                                key=img_features,
                                value=img_features,
                                key_padding_mask=None,
                                incremental_state=incremental_state,
                                static_kv=True,
                                )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual_x + x

        residual_x = x

        # Norm Layer 3
        x = self.layer_norms[2](x)

        # Fully connected Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Fully connected Layer 2
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual_x + x

        # Norm and output
        x = self.last_ln(x)

        return x


"""==================================================================
==========================Transformer================================
===================================================================== """

class Transformer(nn.Module):

    def __init__(self, embed_size, vocab_size, dropout=0.5, seq_len=20,
                 attention_nheads=16, num_layers=8,scale_embed_grad=False):

        super(Transformer, self).__init__()

        self.dropout = dropout
        self.seq_len = seq_len
        self.embed_tokens = nn.Embedding(vocab_size, embed_size, padding_idx=vocab_size-1,
                                         scale_grad_by_freq=scale_embed_grad)

        nn.init.normal_(self.embed_tokens.weight, mean=0, std=embed_size ** -0.5)

        #Prepare the norm layers
        self.layer_norms_in = nn.ModuleList([LayerNorm(embed_size) for i in range(3)])
        self.embed_scale = math.sqrt(embed_size)

        #Add Decoder Layers according the num_layer input
        self.layers = nn.ModuleList([])
        self.layers.extend([
            DecoderLayer(embed_size, attention_nheads, dropout=dropout)
            for i in range(num_layers)
        ])


        self.linear = Linear(embed_size, vocab_size-1)

    def forward(self, captions, img_features, incremental_state=None):

        img_features = img_features.permute(0, 2, 1)
        img_features = img_features.transpose(0, 1)
        self.layer_norms_in[1](img_features)

        if incremental_state is not None:
            captions = captions[:, -1:]

        # Prepare tokens and positions
        x = self.embed_scale * self.embed_tokens(captions)
        x = self.layer_norms_in[2](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)

        #Go through each decoder layer
        for p, layer in enumerate(self.layers):
            x  = layer(x,incremental_state,img_features)
            
        x = x.transpose(0, 1)
        x = self.linear(x)

        _, predicted = x.max(dim=-1)

        return x, predicted

    def sample(self, temperature=1.0, img_features=None, first_token_value=0):

        incremental_state = {}
        fs = img_features.size(0)
        first_word = torch.ones(fs)*first_token_value.to(device).long()
        ingredient_ids = [first_word]
        ingredients_list = []

        for i in range(self.seq_len):

            #Get values correpond to ingredients
            outputs, _ = self.forward(torch.stack(ingredient_ids, 1),
                                      img_features, incremental_state)
            outputs = outputs.squeeze(1)
          
            if i == 0:
                predicted_mask = torch.zeros(outputs.shape).float().to(device)
            else:
                # Ensure no repetitions, for those already selected set to inifinity
                batch_ind = [j for j in range(fs) if ingredient_ids[i][j] != 0]
                ingredient_ids_new = ingredient_ids[i][batch_ind]
                predicted_mask[batch_ind, ingredient_ids_new] = float('-inf')

                # Mask previously selected ingredients
                outputs += predicted_mask

            ingredients_list.append(outputs)
            outputs_prob = torch.nn.functional.softmax(outputs, dim=-1)
            _, predicted = outputs_prob.max(1)
            predicted = predicted.detach()

            ingredient_ids.append(predicted)

        ingredient_ids = torch.stack(ingredient_ids[1:], 1)
        ingredients_list = torch.stack(ingredients_list, 1)

        return ingredient_ids, ingredients_list