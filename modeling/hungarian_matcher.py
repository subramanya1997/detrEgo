# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from modeling.span_utils import generalized_span_iou, span_cw_to_xx


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, span_type, cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.span_type = span_type
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_proposals, 2] with the predicted span coordinates
                 "pred_logits": Tensor of dim [batch_size, num_proposals, num_classes] with the classification logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_proposals, num_target_spans)
        """
        bs, num_proposals = outputs["pred_spans"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_proposals, num_sentences]
        out_spans = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_proposals, 2]

        # Also concat the target labels and spans
        tgt_ids = torch.cat([v["spans"] for v in targets["target_bools"]])  # [total #spans in the batch]\
        tgt_spans = torch.cat([v["spans"] for v in targets["target_spans"]])  # [num_target_spans in batch, 2]
        # tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   # [total #spans in the batch]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_proposals, total #spans in the batch]

        # Compute the L1 cost between spans
        cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_proposals, total #spans in the batch]

        # Compute the giou cost between spans
        # [batch_size * num_proposals, total #spans in the batch]
        if self.span_type == 'cw':
            cost_giou = -generalized_span_iou(span_cw_to_xx(out_spans), span_cw_to_xx(tgt_spans))
        elif self.span_type == 'xx':
            cost_giou = -generalized_span_iou(out_spans, tgt_spans)

        # Final cost matrix
        # import ipdb; ipdb.set_trace()
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_proposals, -1).cpu() # [batch_size, num_proposals, total #spans in the batch]

        sizes = [len(v["spans"]) for v in targets["target_bools"]] # #spans per batch [#spans, #spans, #spans, ...]
        if sum(sizes) != C.shape[2]:
            print(sum(sizes), len(C.split(sizes, -1)), C.shape, tgt_spans.shape, tgt_ids.shape)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] # pairs of (pred, target)

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_hungarian_matcher(args):
    return HungarianMatcher(
        span_type=args.span_type,
        cost_span=args.set_cost_span,
        cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_query,
    )