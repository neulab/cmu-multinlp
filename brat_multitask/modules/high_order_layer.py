from typing import Dict
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.common import FromParams


class HighOrderLayer(torch.nn.Module, FromParams):
    ''' High order interaction for span pairs '''
    def __init__(self,
                 input_dim: int,
                 task2majorlabel: Dict[str, str],
                 num_order: int = 1,
                 task2dummy: Dict[str, bool] = None,
                 vocab: Vocabulary = None) -> None:
        super(HighOrderLayer, self).__init__()
        self.num_order = num_order
        self.task2majorlabel = task2majorlabel
        self.task2dummy = defaultdict(lambda: False)
        if task2dummy:
            self.task2dummy.update(task2dummy)
        self.vocab = vocab
        self.gate_linear = nn.Linear(input_dim * 2, input_dim)


    def forward(self,
                span_repr: torch.FloatTensor,  # SHAPE: (batch_size, num_span1, emb_dim)
                span_pair_ind: torch.LongTensor,  # SHAPE: (batch_size, num_span1, num_span2, 2)
                span_pair_mask: torch.FloatTensor,  # SHAPE: (batch_size, num_span1, num_span2)
                span_pair_layer: torch.nn.Module,
                span_pair_label_proj: torch.nn.Module,
                task: str,
                ) -> torch.Tensor:  # SHAPE: (batch_size, num_span1, emb_dim)
        bs, ns1, ns2, _ = span_pair_ind.size()

        for i in range(self.num_order):
            # compute span pair logits
            # SHAPE: (batch_size, num_span1, num_span2, num_class)
            span_pair_logits = span_pair_label_proj(span_pair_layer(span_repr, span_pair_ind.view(bs, -1, 2)))
            span_pair_logits = span_pair_logits.view(bs, ns1, ns2, -1)
            majorlabel = self.vocab.get_token_index(self.task2majorlabel[task], '{}_span_pair_labels'.format(task))
            # SHAPE: (batch_size, num_span1, num_span2)
            span_pair_logits = span_pair_logits[:, :, :, majorlabel]

            # mask out fake pairs
            span_pair_logits = span_pair_logits + span_pair_mask.log()
            if self.task2dummy[task]:
                # add a dummy class. This is important to avoid nan bug
                span_pair_logits = torch.cat([torch.zeros_like(span_pair_logits[:, :, :1]), span_pair_logits], -1)

            # compute distribution over num_span2 dim
            # SHAPE: (batch_size, num_span1, num_span2)
            span_pair_prob = F.softmax(span_pair_logits, dim=-1)
            if self.task2dummy[task]:
                span_pair_prob = span_pair_prob[:, :, 1:]
            # if all the elements are maked, nan appears. replace nan with zero
            #span_pair_prob = torch.where(torch.isnan(span_pair_prob), torch.zeros_like(span_pair_prob), span_pair_prob)
            # SHAPE: (batch_size, num_span1, num_span1)
            span_pair_prob_all = torch.zeros((bs, ns1, ns1), device=span_repr.device)
            # use scatter_add_ for padding
            span_pair_prob_all.scatter_add_(-1, span_pair_ind[:, :, :, 1], span_pair_prob)

            # sum over num_span2
            # SHAPE: (batch_size, num_span1, emb_dim)
            span_repr_sum = torch.matmul(span_pair_prob_all, span_repr)

            # update span repr
            # SHAPE: (batch_size, num_span1, emb_dim)
            gate = torch.sigmoid(self.gate_linear(torch.cat([span_repr, span_repr_sum], -1)))
            span_repr = span_repr_sum * gate + span_repr * (1 - gate)

        return span_repr
