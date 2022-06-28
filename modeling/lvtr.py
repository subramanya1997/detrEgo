# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import os
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns

from modeling.transformer_joint import build_transformer_joint
from modeling.position_encoding import build_position_encoding


class LVTR(nn.Module):
    """ End-to-End Video Grounding with Transformer """

    def __init__(self, transformer, vid_position_embed, txt_position_embed,
                 txt_dim, vid_dim, num_proposals, input_dropout, aux_loss=False,
                 n_input_proj=2, n_class_head=2,
                 num_classes=1, pred_label='att'):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            vid_position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_proposals: number of proposals per sentence, ie detection slot.
                         (num_proposals)x(num_sentences) is the maximal number of objects LVTR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_proposals = num_proposals
        self.num_classes = num_classes
        self.transformer = transformer
        self.vid_position_embed = vid_position_embed
        self.txt_position_embed = txt_position_embed
        self.hidden_dim = transformer.d_model
        self.span_embed = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
        self.n_input_proj = n_input_proj
        self.pred_label = pred_label
        self.class_head = nn.Linear(self.hidden_dim, num_classes)

        self.query_embed = nn.Embedding(num_proposals, self.hidden_dim)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, self.hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, self.hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.aux_loss = aux_loss

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, epoch_i=None, idx=None, save_dir=None, scope="train"):
        """The forward expects two tensors:
                - src_txt: [batch_size, L_txt, D_txt]
                - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
                - src_vid: [batch_size, L_vid, D_vid]
                - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
            It returns a dict with the following elements:
                - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_proposals x (num_classes + 1)]
                - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
                - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)

        pos_vid = self.vid_position_embed(src_vid, src_vid_mask) 
        pos_txt = self.txt_position_embed(src_txt, src_txt_mask)

        src = torch.cat([src_vid, src_txt], dim=1)  # (batch_size, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (batch_size, L_vid+L_txt)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        if src.shape[1] >= self.hidden_dim:
            print(f"{idx} :  {src.shape[1]}")

        hs, memory, att = self.transformer(
            src, ~mask, self.query_embed.weight, pos
        )  # hs: (#layers, batch_size, #queries, d)
        txt_mem = memory[:, src_vid.shape[1]:]  # (batch_size, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (batch_size, L_vid, d)
        vid_att = [att_[:, :, :-self.num_classes] for att_ in att]
        txt_att = [att_[:, :, -self.num_classes:] for att_ in att]

        if self.pred_label == 'att':
            outputs_class = txt_att
        elif self.pred_label == 'sim':
            outputs_class = torch.stack([hs_ @ txt_mem.transpose(1, 2) for hs_ in hs]) # lx[(BxQxD)x(BxDxL)] = lxBxQxL
        elif self.pred_label == 'cos':
            outputs_class = torch.stack([F.normalize(hs_) @ F.normalize(txt_mem.transpose(1, 2)) for hs_ in hs]) # lx[(BxQxD)x(BxDxL)] = lxBxQxL
        elif self.pred_label == 'pred':
            outputs_class = self.class_head(hs)
        else:
            raise NotImplementedError

        outputs_coord = self.span_embed(hs)  # (#layers, batch_size, #queries, 2)
        outputs_coord = outputs_coord.sigmoid()
        out = {
            'pred_logits': outputs_class[-1],
            'pred_spans': outputs_coord[-1]
        }

        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        # attention visualize
        if False:
            save_dir = os.path.join(save_dir, f'epoch_{str(epoch_i)}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            vidatt = vid_att[-1].detach().cpu() # (#layers, batch_size, #queries, #frames)
            txtatt = txt_att[-1].detach().cpu() # (#layers, batch_size, #queries, #sentences)
            for i, (vatt, tatt) in enumerate(zip(vidatt, txtatt)):
                vatt = torch.vstack((vatt[:14, :], vatt[29:, :], vatt[14:28, :]))

                att_path = os.path.join(save_dir, f'attention')
                if not os.path.exists(att_path):
                    os.makedirs(att_path)

                ax = sns.heatmap(vatt, cmap="YlGnBu", xticklabels=False, yticklabels=False, cbar=False, square=True)
                plt.savefig(f'{att_path}/vid_att_{scope}_{str(idx)}.png')
                
                ax = sns.heatmap(tatt, cmap="YlGnBu", xticklabels=False, yticklabels=False, cbar=False, square=True)
                plt.savefig(f'{att_path}/txt_att_{scope}_{str(idx)}.png')

        # correspondence visualize
        
            att = outputs_class[-1][0].detach().cpu()
            averaged_att = []
            for i in range(att.shape[1]):
                averaged_att.append(att[i*10:(i+1)*10].mean(0))
            averaged_att = torch.stack(averaged_att)
            ax = sns.heatmap(averaged_att, xticklabels=False, yticklabels=False, cbar=False, square=True)
            
            correspondence_path = os.path.join(save_dir, f'correspondence')
            if not os.path.exists(correspondence_path):
                os.makedirs(correspondence_path)
            plt.savefig(f'{correspondence_path}/{scope}_{str(idx)}.png')

        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_head(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).

    transformer = build_transformer_joint(args)    
    vid_pos_embed, txt_pos_embed = build_position_encoding(args)

    return LVTR(
        transformer,
        vid_pos_embed,
        txt_pos_embed,
        txt_dim=args.txt_feat_dim,
        vid_dim=args.vid_feat_dim,
        num_proposals=args.num_querys * 10,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        n_input_proj=args.n_input_proj,
        num_classes= args.num_querys + 1,
        pred_label=args.pred_label
    )