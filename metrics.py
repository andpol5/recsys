import random
from math import log2
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


def ndcg_score(y_true, y_score):
    rating_pairs = np.stack([y_true, y_score], axis=1).tolist()
    sorted_pairs = sorted(rating_pairs, key=lambda x: x[1], reverse=True)
    dcg = sum(
        (true_rating / log2(index + 2))
        for index, (true_rating, _) in enumerate(sorted_pairs)
    )
    ideal_pairs = sorted(rating_pairs, key=lambda x: x[0], reverse=True)
    idcg = sum(
        (true_rating / log2(index + 2))
        for index, (true_rating, _) in enumerate(ideal_pairs)
    )
    ndcg = dcg / idcg
    return ndcg


def novelty_score(predicted: List[int], pop: List[int], num_users: int, num_items: int):
    mean_self_information = []
    k = 0
    for sublist in predicted:
        self_information = 0
        k += 1
        for i in sublist:
            if pop[i] > 0:
                self_information += np.sum(-np.log2(pop[i] / num_users))
            else:
                continue
        mean_self_information.append(self_information / num_items)
    novelty = sum(mean_self_information) / k
    return novelty


def prediction_coverage_score(predicted: List[list], catalog: list):
    predicted_flattened = [p for sublist in predicted for p in sublist]
    unique_items_pred = set(predicted_flattened)

    num_unique_predictions = len(unique_items_pred)
    prediction_coverage = round(num_unique_predictions / (len(catalog) * 1.0) * 100, 2)
    return prediction_coverage


def catalog_coverage_score(predicted: List[list], catalog: list, k: int) -> float:
    sampling = random.choices(predicted, k=k)
    predicted_flattened = [p for sublist in sampling for p in sublist]
    L_predictions = len(set(predicted_flattened))
    catalog_coverage = round(L_predictions / (len(catalog) * 1.0) * 100, 2)
    return catalog_coverage


def personalization_score(predicted: np.ndarray) -> float:
    similarity = cosine_similarity(X=predicted, dense_output=False)
    dim = similarity.shape[0]
    personalization = (similarity.sum() - dim) / (dim * (dim - 1))
    return 1 - personalization
