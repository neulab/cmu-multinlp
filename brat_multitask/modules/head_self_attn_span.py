import torch
import torch.nn as nn
from overrides import overrides
import copy

from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.nn import util


class BertSelfAttnLayers(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for layer in layers])


    def forward(self,
                hidden_states: torch.FloatTensor,  # SHAPE: (batch_size, seq_len, hidden_size)
                attention_mask: torch.LongTensor):  # SHAPE: (batch_size, seq_len)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.float()
        attention_mask = (1.0 - attention_mask) * -10000.0
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


@SpanExtractor.register('head_self_attentive')
class HeadSelfAttentiveSpanExtractor(SpanExtractor):
    """
    Computes span representations by generating an unnormalized attention score for each
    word in the document. Spans representations are computed with respect to these
    scores by normalising the attention scores for words inside the span.

    Given these attention distributions over every span, this module weights the
    corresponding vector representations of the words in the span by this distribution,
    returning a weighted representation of each span.

    Parameters
    ----------
    input_dim : ``int``, required.
        The final dimension of the ``sequence_tensor``.

    Returns
    -------
    attended_text_embeddings : ``torch.FloatTensor``.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """
    def __init__(self,
                 input_dim: int,
                 num_head: int = 3,
                 bert_self_attn_layers: BertSelfAttnLayers = None) -> None:
        super().__init__()
        self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))
        self._num_heads = num_head
        self._span_token_emb = nn.Parameter(torch.Tensor(input_dim))
        nn.init.normal_(self._span_token_emb)
        if bert_self_attn_layers is not None:
            self._input_dim = self._output_dim = input_dim
            self._stacked_self_attention = bert_self_attn_layers
        else:
            self._input_dim = input_dim
            self._output_dim = input_dim // 4
            self._stacked_self_attention = StackedSelfAttentionEncoder(
                input_dim=input_dim,
                hidden_dim=self._output_dim,
                projection_dim=self._output_dim,
                feedforward_hidden_dim=4 * self._output_dim,
                num_layers=2,
                num_attention_heads=1,
                use_positional_encoding=True)


    def get_input_dim(self) -> int:
        return self._input_dim


    def get_output_dim(self) -> int:
        return self._output_dim


    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:
        batch_size, num_spans = span_indices.size()[:2]

        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # shape (batch_size, sequence_length, 1)
        global_attention_logits = self._global_attention(sequence_tensor)

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = util.get_range_vector(max_batch_span_width,
                                                       util.get_device_of(sequence_tensor)).view(1, 1, -1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = (max_span_range_indices <= span_widths).float()
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

        # compute span head weight
        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = util.batched_index_select(global_attention_logits,
                                                          span_indices,
                                                          flat_span_indices).squeeze(-1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_head_weights = util.masked_softmax(span_attention_logits, span_mask)

        # get head words indices
        top_num_heads = min(self._num_heads, max_batch_span_width)
        # Shape: (batch_size, num_spans, num_heads)
        span_head_ind = span_head_weights.topk(top_num_heads, -1)[1]
        # make sure the index is consistent with the original order
        span_head_ind = torch.sort(span_head_ind, -1)[0]
        # Shape: (batch_size * num_spans * num_heads)
        flat_span_head_ind = util.flatten_and_batch_shift_indices(
            span_head_ind.view(-1, top_num_heads), max_batch_span_width)

        # select emb and mask
        # Shape: (batch_size, num_spans, num_heads)
        span_head_ind_external = util.batched_index_select(
            span_indices.view(-1, max_batch_span_width, 1),
            span_head_ind.view(-1, top_num_heads),
            flat_span_head_ind).view(*span_head_ind.size())
        # Shape: (batch_size, num_spans, num_heads)
        span_head_mask = util.batched_index_select(
            span_mask.view(-1, max_batch_span_width, 1),
            span_head_ind.view(-1, top_num_heads),
            flat_span_head_ind).view(*span_head_ind.size())
        # Shape: (batch_size, num_spans, num_heads, emb_dim)
        span_head_emb = util.batched_index_select(sequence_tensor, span_head_ind_external, flattened_indices=None)

        # concat with span token
        # Shape: (batch_size, num_spans, 1, emb_dim)
        span_token_emb = self._span_token_emb.view(1, 1, 1, -1).expand(batch_size, num_spans, -1, -1)
        # Shape: (batch_size, num_spans, num_heads + 1, emb_dim)
        span_head_emb = torch.cat([span_head_emb, span_token_emb], 2)
        # Shape: (batch_size, num_spans, num_heads + 1, emb_dim)
        span_head_mask = nn.ConstantPad1d((1, 0), 1)(span_head_mask)

        # self attention span representation
        span_head_emb = self._stacked_self_attention(
            span_head_emb.view(-1, top_num_heads + 1, self._input_dim),
            span_head_mask.view(-1, top_num_heads + 1))

        # aggregate
        # Shape: (batch_size, num_spans, num_heads + 1, emb_dim)
        span_head_emb = span_head_emb.view(batch_size, num_spans, top_num_heads + 1, self._output_dim)
        # Shape: (batch_size, num_spans, emb_dim)
        span_embeddings = span_head_emb[:, :, 0]

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)
        #attended_text_embeddings = util.weighted_sum(span_embeddings, span_attention_weights)

        if span_indices_mask is not None:
            # Above we were masking the widths of spans with respect to the max
            # span width in the batch. Here we are masking the spans which were
            # originally passed in as padding.
            return span_embeddings * span_indices_mask.unsqueeze(-1).float()

        return span_embeddings
