from typing import Iterable
import numpy as np


def pair_accuracy(list1: Iterable[float], list2: Iterable[float]) -> float:
    if len(list1) != len(list2):
        raise ValueError("Both lists must be of the same length")

    total_pairs = 0
    correct_pairs = 0

    for i in range(len(list1)):
        for j in range(i + 1, len(list1)):
            total_pairs += 1

            if (list1[i] > list1[j] and list2[i] > list2[j]) or (
                list1[i] < list1[j] and list2[i] < list2[j]
            ):
                correct_pairs += 1
            elif list1[i] == list1[j] and list2[i] == list2[j]:
                correct_pairs += 1

    pair_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
    return pair_accuracy


def average_pair_accuracy(
    query_ids: Iterable[int], list1: Iterable[float], list2: Iterable[float]
) -> float:
    unique_query_ids = np.unique(query_ids)
    pair_accuracies = []

    for query_id in unique_query_ids:
        indices = np.where(query_ids == query_id)[0]
        query_list1 = np.array(list1)[indices]
        query_list2 = np.array(list2)[indices]
        pair_accuracy_value = pair_accuracy(query_list1, query_list2)
        pair_accuracies.append(pair_accuracy_value)
    return np.mean(pair_accuracies)


def dcg_at_k(scores, k):
    """Calculate DCG for the given scores and rank k."""
    return np.sum((2**scores - 1) / np.log2(np.arange(2, k + 2)))


def ndcg_at_k(true_scores, predicted_scores, k):
    """Calculate nDCG for the given true scores and predicted scores at rank k."""
    order = np.argsort(predicted_scores)[::-1]
    true_scores_sorted = np.take(true_scores, order)
    DCG = dcg_at_k(true_scores_sorted, k)
    IDCG = dcg_at_k(np.sort(true_scores)[::-1], k)
    return DCG / IDCG if IDCG > 0 else 0


def calculate_ndcg(query_ids, predicted_scores, true_scores):
    """Calculate the average nDCG score for multiple queries."""
    unique_queries = np.unique(query_ids)
    ndcg_scores = []

    for query in unique_queries:
        indices = np.where(query_ids == query)
        ndcg = ndcg_at_k(
            np.array(true_scores)[indices],
            np.array(predicted_scores)[indices],
            k=indices[0].size,
        )
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)
