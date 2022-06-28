import torch
import torch.nn as nn
from modeling.lvtr import build_head

class MEME(nn.Module):

    def __init__(self, head):
        super(MEME, self).__init__()
        self.head = head
    
    def forward(self, video, query, video_mask = None, query_mask = None,
                epoch_i=None, idx=None, save_dir=None, scope="Train"):

        outputs = self.head(query, query_mask, video, video_mask, epoch_i, idx, save_dir, scope)
        
        return outputs

def build_model(args):
    
    head = build_head(args)
    model = MEME(head)
    # print(model)
    return model