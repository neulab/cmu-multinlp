from typing import Any, Dict, List, Tuple

from overrides import overrides
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.conll_coref_scores import ConllCorefScores, Scorer


@Metric.register('my_conll_coref_scores')
class MyConllCorefScores(ConllCorefScores):
    def __init__(self,
                 use_same_antecedent_indices: bool = False) -> None:
        super(MyConllCorefScores, self).__init__()
        self.scorers = [MyScorer(m) for m in (MyScorer.muc, MyScorer.b_cubed, MyScorer.ceafe)]
        self.use_same_antecedent_indices = use_same_antecedent_indices


    @overrides
    def __call__(self,  # type: ignore
                 top_spans: torch.Tensor,
                 antecedent_indices: torch.Tensor,  # (batch_size, num_spans, num_antecedents)
                 predicted_antecedents: torch.Tensor,
                 metadata_list: List[Dict[str, Any]]):
        top_spans, antecedent_indices, predicted_antecedents = self.unwrap_to_tensors(top_spans,
                                                                                      antecedent_indices,
                                                                                      predicted_antecedents)
        for i, metadata in enumerate(metadata_list):
            gold_clusters, mention_to_gold = self.get_gold_clusters(metadata["clusters"])
            if self.use_same_antecedent_indices:
                predicted_clusters, mention_to_predicted = self.get_predicted_clusters(
                    top_spans[i], antecedent_indices, predicted_antecedents[i])
            else:
                predicted_clusters, mention_to_predicted = self.get_predicted_clusters(
                    top_spans[i], antecedent_indices[i], predicted_antecedents[i])
            for scorer in self.scorers:
                scorer.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)


    @staticmethod
    def get_predicted_clusters(top_spans: torch.Tensor,
                               antecedent_indices: torch.Tensor,
                               predicted_antecedents: torch.Tensor) -> Tuple[List[Tuple[Tuple[int, int], ...]],
                                                                             Dict[Tuple[int, int],
                                                                                  Tuple[Tuple[int, int], ...]]]:
        # Pytorch 0.4 introduced scalar tensors, so our calls to tuple() and such below don't
        # actually give ints unless we convert to numpy first.  So we do that here.
        top_spans = top_spans.numpy()  # (num_spans, 2)
        antecedent_indices = antecedent_indices.numpy()  # (num_spans, num_antecedents)
        predicted_antecedents = predicted_antecedents.numpy()  # (num_spans,)

        predicted_clusters_to_ids: Dict[Tuple[int, int], int] = {}
        clusters: List[List[Tuple[int, int]]] = []
        for i, predicted_antecedent in enumerate(predicted_antecedents):
            if predicted_antecedent < 0:
                continue

            # Find predicted index in the antecedent spans.
            predicted_index = antecedent_indices[i, predicted_antecedent]
            # Must be a previous span.
            antecedent_span: Tuple[int, int] = tuple(top_spans[predicted_index])  # type: ignore

            # Check if we've seen the span before.
            if antecedent_span in predicted_clusters_to_ids.keys():
                predicted_cluster_id: int = predicted_clusters_to_ids[antecedent_span]
            else:
                # We start a new cluster.
                predicted_cluster_id = len(clusters)
                clusters.append([antecedent_span])
                predicted_clusters_to_ids[antecedent_span] = predicted_cluster_id

            mention: Tuple[int, int] = tuple(top_spans[i])  # type: ignore
            clusters[predicted_cluster_id].append(mention)
            predicted_clusters_to_ids[mention] = predicted_cluster_id

        # finalise the spans and clusters.
        final_clusters = [tuple(cluster) for cluster in clusters]
        # Return a mapping of each mention to the cluster containing it.
        mention_to_cluster: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {
            mention: final_clusters[cluster_id]
            for mention, cluster_id in predicted_clusters_to_ids.items()
        }

        return final_clusters, mention_to_cluster


    @overrides
    def reset(self):
        self.scorers = [MyScorer(metric) for metric in (MyScorer.muc, MyScorer.b_cubed, MyScorer.ceafe)]


class MyScorer(Scorer):
    def __init__(self, metric):
        super(MyScorer, self).__init__(metric)


    @staticmethod
    def ceafe(clusters, gold_clusters):
        """
        Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.
        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        """
        clusters = [cluster for cluster in clusters if len(cluster) != 1]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = Scorer.phi4(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])
        return similarity, len(clusters), similarity, len(gold_clusters)
