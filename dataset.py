import numpy as np
import pandas as pd
import torch.utils.data
from enum import IntEnum
from numpy.random import choice, randint


class RatingFormat(IntEnum):
    BINARY = 1
    RATING = 2


class MovieLens20MDataset(torch.utils.data.Dataset):
    """
    MovieLens 20M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    # tags.csv:
    # userId,movieId,tag,timestamp

    # ratings.csv:
    # userId,movieId,tag,timestamp

    # movies.csv:
    # movieId,title,genres

    # genome-tags.csv:
    # tagId,tag

    # genome-scores.csv:
    # movieId,tagId,relevance

    def __init__(self, dataset_path: str, return_format: RatingFormat, negative_sample_threshold: float = 3, num_negative_samples: int = 4):
        data = pd.read_csv(
            dataset_path, sep=",", engine="c", header="infer"
        ).to_numpy()[:, :3]
        self.no_users = np.max(data[:, 0].astype(np.int64)) + 1
        self.no_movies = np.max(data[:, 1].astype(np.int64)) + 1
        self.no_samples = len(data)
        self.num_negative_samples = num_negative_samples

        self.positive_ratings = data[np.where(data[:, 2] >= negative_sample_threshold, 1, 0)]
        self.negative_ratings = data[np.where(data[:, 2] < negative_sample_threshold, 1, 0)]

        pos_ratings_mask = np.zeros(len(self.positive_ratings), dtype=np.int64)
        pos_ratings_mask[::num_negative_samples] = 1
        
        self.return_format = return_format
        self.neg_threshold = negative_sample_threshold

    def __len__(self):
        return self.no_samples

    def __getitem__(self, index):
        sample_idx = randint(0, len(self.positive_ratings)) if index % self.num_negative_samples == 0 else randint(0, len(self.negative_ratings))
        sample = self.positive_ratings[sample_idx] if index % self.num_negative_samples == 0 else self.negative_ratings[sample_idx]
        rating = sample[2].astype(np.float32)
        if self.return_format == RatingFormat.BINARY:
            rating = rating >= self.neg_threshold
        return sample[0].astype(np.int64), sample[1].astype(np.int64), rating
