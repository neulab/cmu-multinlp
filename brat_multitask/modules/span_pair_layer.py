from typing import Union, List
import copy
import torch
import torch.nn as nn

from allennlp.common import FromParams
from allennlp.nn import util
from allennlp.modules import FeedForward, TimeDistributed
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules.token_embedders import Embedding


class SpanPairLayer(torch.nn.Module, FromParams):
    ''' Represent span pairs '''
    def __init__(self,
                 dim_reduce_layer: FeedForward = None,
                 separate: bool = False,
                 repr_layer: FeedForward = None) -> None:
        super(SpanPairLayer, self).__init__()

        self.inp_dim, self.out_dim = None, None

        self.dim_reduce_layer1 = self.dim_reduce_layer2 = dim_reduce_layer
        if dim_reduce_layer is not None:
            self.inp_dim = self.inp_dim or dim_reduce_layer.get_input_dim()
            self.out_dim = dim_reduce_layer.get_output_dim()
            self.dim_reduce_layer1 = TimeDistributed(dim_reduce_layer)
            if separate:
                self.dim_reduce_layer2 = copy.deepcopy(self.dim_reduce_layer1)
            else:
                self.dim_reduce_layer2 = self.dim_reduce_layer1

        self.repr_layer = None
        if repr_layer is not None:
            self.inp_dim = self.inp_dim or repr_layer.get_input_dim()
            self.out_dim = repr_layer.get_output_dim()
            self.repr_layer = TimeDistributed(repr_layer)


    def get_output_dim(self):
        return self.out_dim


    def get_input_dim(self):
        return self.inp_dim


    def forward(self,
                span1: torch.Tensor,  # SHAPE: (batch_size, num_span1, span_dim)
                span2: torch.Tensor  # SHAPE: (batch_size, num_span2, span_dim)
                ):

        if self.dim_reduce_layer1 is not None:
            span1 = self.dim_reduce_layer1(span1)
        if self.dim_reduce_layer2 is not None:
            span2 = self.dim_reduce_layer2(span2)

        num_span1 = span1.size(1)
        num_span2 = span2.size(1)
        span_dim = span1.size(2)

        if self.repr_layer is not None:
            # SHAPE: (batch_size, num_span1, num_span2, span_dim * 2)
            span_pairs = torch.cat([span1.unsqueeze(2).repeat(1, 1, num_span2, 1),
                                    span2.unsqueeze(1).repeat(1, num_span1, 1, 1)], -1)
            span_pairs = span_pairs.view(-1, num_span1 * num_span2, span_dim * 2)
            # SHAPE: (batch_size, num_span1 * num_span2, label_dim)
            output = self.repr_layer(span_pairs)
            return output

        return span1, span2


class SpanPairPairedLayer(torch.nn.Module, FromParams):
    ''' Represent span pairs '''
    def __init__(self,
                 dim_reduce_layer: FeedForward = None,
                 separate: bool = False,
                 repr_layer: FeedForward = None,
                 pair: bool = True,
                 combine: str = 'concat',
                 dist_emb_size: int = None) -> None:
        super(SpanPairPairedLayer, self).__init__()

        self.inp_dim, self.out_dim = None, None
        self.pair = pair
        self.combine = combine
        assert combine in {'concat', 'coref'}  # 'coref' means using concat + dot + width
        if combine == 'coref':
            self.num_distance_buckets = 10
            self.distance_embedding = Embedding(self.num_distance_buckets, dist_emb_size)

        self.dim_reduce_layer1 = self.dim_reduce_layer2 = dim_reduce_layer
        if dim_reduce_layer is not None:
            self.inp_dim = self.inp_dim or dim_reduce_layer.get_input_dim()
            self.out_dim = dim_reduce_layer.get_output_dim()
            self.dim_reduce_layer1 = TimeDistributed(dim_reduce_layer)
            if separate:
                self.dim_reduce_layer2 = copy.deepcopy(self.dim_reduce_layer1)
            else:
                self.dim_reduce_layer2 = self.dim_reduce_layer1
            if pair:
                self.out_dim *= 2

        self.repr_layer = None
        if repr_layer is not None:
            if not pair:
                raise Exception('MLP needs paired input')
            self.inp_dim = self.inp_dim or repr_layer.get_input_dim()
            self.out_dim = repr_layer.get_output_dim()
            self.repr_layer = TimeDistributed(repr_layer)


    def get_output_dim(self):
        return self.out_dim


    def get_input_dim(self):
        return self.inp_dim


    def forward(self,
                span: torch.Tensor,  # SHAPE: (batch_size, num_spans, span_dim)
                span_pairs: torch.LongTensor  # SHAPE: (batch_size, num_span_pairs)
                ):
        span1 = span2 = span
        if self.dim_reduce_layer1 is not None:
            span1 = self.dim_reduce_layer1(span)
        if self.dim_reduce_layer2 is not None:
            span2 = self.dim_reduce_layer2(span)

        if not self.pair:
            return span1, span2

        num_spans = span.size(1)

        # get span pair embedding
        span_pairs_p = span_pairs[:, :, 0]
        span_pairs_c = span_pairs[:, :, 1]
        # SHAPE: (batch_size * num_span_pairs)
        flat_span_pairs_p = util.flatten_and_batch_shift_indices(span_pairs_p, num_spans)
        flat_span_pairs_c = util.flatten_and_batch_shift_indices(span_pairs_c, num_spans)
        # SHAPE: (batch_size, num_span_pairs, span_dim)
        span_pair_p_emb = util.batched_index_select(span1, span_pairs_p, flat_span_pairs_p)
        span_pair_c_emb = util.batched_index_select(span2, span_pairs_c, flat_span_pairs_c)
        if self.combine == 'concat':
            # SHAPE: (batch_size, num_span_pairs, span_dim * 2)
            span_pair_emb = torch.cat([span_pair_p_emb, span_pair_c_emb], -1)
        elif self.combine == 'coref':
            # use the indices gap as distance, which requires the indices to be consistent
            # with the order they appear in the sentences
            distance = span_pairs_p - span_pairs_c
            # SHAPE: (batch_size, num_span_pairs, dist_emb_dim)
            distance_embeddings = self.distance_embedding(
                util.bucket_values(distance, num_total_buckets=self.num_distance_buckets))
            # SHAPE: (batch_size, num_span_pairs, span_dim * 3)
            span_pair_emb = torch.cat([span_pair_p_emb,
                                       span_pair_c_emb,
                                       span_pair_p_emb * span_pair_c_emb,
                                       distance_embeddings], -1)

        if self.repr_layer is not None:
            # SHAPE: (batch_size, num_span_pairs, out_dim)
            span_pair_emb = self.repr_layer(span_pair_emb)

        return span_pair_emb


class SpanPairLabelProjectionLayer(torch.nn.Module):
    ''' Map span pair representation to label logits '''
    def __init__(self,
                 inp_dim: int,
                 label_dim: int,
                 method: str) -> None:
        super(SpanPairLabelProjectionLayer, self).__init__()
        assert method in {'mlp', 'biaffine', 'biaffine_paired'}
        self.method = method
        if method == 'mlp':
            self.label_proj = TimeDistributed(nn.Linear(inp_dim, label_dim))
        elif method == 'biaffine':
            self.label_proj = BilinearMatrixAttention(
                inp_dim, inp_dim, use_input_biases=True, label_dim=label_dim)
        elif method == 'biaffine_paired':
            # exactly the same as the last one but used with paired inputs
            self.bilinear = nn.modules.Bilinear(inp_dim // 2, inp_dim // 2, label_dim, bias=True)
            self.linear = TimeDistributed(nn.Linear(inp_dim, label_dim, bias=False))
            self.label_proj = lambda x: self.bilinear(*x.split(inp_dim // 2, dim=-1)) + self.linear(x)


    def forward(self, span_repr: Union[torch.Tensor, List[torch.Tensor]]) \
            -> torch.Tensor:  # SHAPE: (batch_size, num_span1 * num_span2, label_dim)
        if self.method == 'mlp':
            logits = self.label_proj(span_repr)
        elif self.method == 'biaffine':
            span1_repr, span2_repr = span_repr
            logits = self.label_proj(span1_repr, span2_repr)
            logits = logits.permute(0, 2, 3, 1)
            logits = logits.view(logits.size(0), -1, logits.size(-1)).contiguous()
        elif self.method == 'biaffine_paired':
            logits = self.label_proj(span_repr)
        else:
            raise NotImplementedError
        return logits
