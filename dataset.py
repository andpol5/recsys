import os
import sys
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.random import choice, randint
from tabulate import tabulate
from torch.utils.data import Dataset



class RatingFormat(IntEnum):
    BINARY = 1
    RATING = 2


class MovieLens20MDataset(Dataset):
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

    def __init__(
        self,
        dataset_path: str,
        return_format: RatingFormat,
        max_rows: int = sys.maxsize,
        max_users: int = None,
    ):
        dataset_path = Path(dataset_path)
        self.emb_columns: List[str] = ["userId", "movieId"]
        self.pred_column: str = "rating"

        ratings_data = pd.read_csv(dataset_path / "ratings.csv", sep=",", header="infer", nrows=max_rows).dropna()
        genres_data = pd.read_csv(dataset_path / "movies.csv", sep=",", engine="pyarrow", header="infer")
        primary_genre_per_movie = genres_data["genres"].str.split("|").str[0]
        self.movie_genres = pd.concat([genres_data["movieId"], primary_genre_per_movie])

        self.feature_sizes: List[int] = [ratings_data[x].max() + 1 for x in self.emb_columns]
        self.movie_data = pd.read_csv(dataset_path / "movies.csv", sep=",", engine="pyarrow", header="infer")

        self.ratings_data = ratings_data
        self.neg_threshold = 2.5

        if max_users is not None:
            first_n_users = ratings_data["userId"].unique()[:max_users]
            ratings_from_first_n_users = ratings_data[
                ratings_data["userId"].isin(first_n_users)
            ]
            ratings_data = ratings_from_first_n_users

        no_users = self.ratings_data["userId"].max()
        no_movies = self.ratings_data["movieId"].max()
        self.no_samples = self.ratings_data.shape[0]
        print(f"Number of users: {no_users} | Number of movies: {no_movies} | Number of samples: {self.no_samples}")
        self.return_format = return_format

    def display_recommendation_output(self, user_id: int, pred_ids: np.ndarray, true_ids: np.ndarray):
        """Returns a dictionary of movie names and genres for a batch of movie IDs"""
        pred_data = self.movie_data.iloc[pred_ids]
        true_data = self.movie_data.iloc[true_ids]
        print(f"predictions for user {user_id}:")
        print(tabulate(pred_data[["title", "genres"]], headers="keys", tablefmt="psql"))
        print(f"ground truth for user {user_id}:")
        print(tabulate(true_data[["title", "genres"]], headers="keys", tablefmt="psql"))

    def __len__(self):
        return self.no_samples

    def __getitem__(self, index):
        sample = self.ratings_data.iloc[index]
        rating = sample["rating"].astype(np.float32)
        if self.return_format == RatingFormat.BINARY:
            rating = (rating >= self.neg_threshold).astype(np.float32)
        features = sample[self.emb_columns].to_numpy()
        # print('index', index, 'features', features, 'rating', rating)
        return features, rating


class CriteoDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        all_data = pd.read_csv("datasets/criteo_1m.txt")

        self.labels = torch.tensor(all_data["label"].values)
        self.features = torch.tensor(all_data.iloc[:, 1:].values)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class DatasetSource(IntEnum):
    MOVIELENS = 1
    AMAZON = 2
    CRITEO = 3


datasets_dict = {
    DatasetSource.CRITEO: CriteoDataset,
    DatasetSource.MOVIELENS: MovieLens20MDataset,
}
