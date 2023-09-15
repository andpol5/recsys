from collections import defaultdict
from enum import IntEnum
from math import log2
from typing import List, Optional
import pandas as pd

pd.options.display.float_format = "{:.2f}".format

import fire
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from dataset import MovieLens20MDataset, RatingFormat
from metrics import *
from models import *
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)


class DatasetSource(IntEnum):
    MOVIELENS = 1
    AMAZON = 2
    CRITEO = 3


class Params:
    learning_rate: int = 5e-4
    weight_decay: float = 1e-5
    layers: List[int] = [64, 32, 16, 8]

    # only used for MF
    embedding_dim: int = 32
    dropout: float = 0.2
    batch_size: int = 128
    eval_size: int = 100
    max_rows: int = 100000
    model_architecture: ModelArchitecture = ModelArchitecture.MATRIX_FACTORIZATION
    dataset_source: DatasetSource = DatasetSource.MOVIELENS
    rating_format: RatingFormat = RatingFormat.BINARY
    max_users: Optional[int] = None


class RecommenderModule(nn.Module):
    def __init__(self, recommender: nn.Module, use_wandb: bool):
        super().__init__()
        self.recommender = recommender
        if Params.rating_format == RatingFormat.BINARY:
            self.loss_fn = torch.nn.BCELoss()
        else:
            self.loss_fn = torch.nn.MSELoss()
        self.use_wandb = use_wandb

    def training_step(self, batch):
        users, items, ratings = batch
        preds = self.recommender(users, items)
        loss = self.loss_fn(preds, ratings)
        if self.use_wandb:
            wandb.log({"train_loss": loss})
        return loss

    def eval_step(self, dataset: MovieLens20MDataset, batch, k: int = 10):
        with torch.no_grad():
            users, items, ratings = batch
            max_user_id = users.max() + 1
            preds = self.recommender(users, items)
            eval_loss = self.loss_fn(preds, ratings).item()
            user_item_ratings = np.empty((max_user_id, k))
            true_item_ratings = np.empty((max_user_id, k))
            for i, user_id in enumerate(users):
                user_id = user_id.item()
                # predict every item for every user
                user_ids = torch.full_like(items, user_id)
                user_preds = self.recommender(user_ids, items)
                top_k_preds = torch.topk(user_preds, k=k).indices
                user_item_ratings[user_id] = top_k_preds.numpy()

                true_top_k = torch.topk(ratings, k=k).indices
                true_item_ratings[user_id] = true_top_k.numpy()
                if i == 0:
                    dataset.display_recommendation_output(
                        user_id, top_k_preds, true_top_k
                    )

            unique_item_catalog = list(set(items.tolist()))
            item_popularity = defaultdict(int)
            for item in items:
                item_popularity[item.item()] += 1

            num_users = len(list(set(users.tolist())))
            num_items = len(list(set(items.tolist())))

            novelty = novelty_score(
                user_item_ratings, item_popularity, num_users, num_items
            )

            user_rating_preds = np.array(
                [p for sublist in user_item_ratings for p in sublist]
            )
            user_rating_ref = np.array(
                [p for sublist in user_item_ratings for p in sublist]
            )

            prediction_coverage = prediction_coverage_score(
                user_item_ratings, unique_item_catalog
            )
            catalog_coverage = catalog_coverage_score(
                user_item_ratings, unique_item_catalog, k
            )

            personalization = personalization_score(user_item_ratings)

            ref_bool, preds_bool = user_rating_ref.astype(bool), user_rating_preds.astype(bool)
            # Handle the case where all values are T or F
            if len(np.unique(ref_bool)) == 2 and len(np.unique(preds_bool)) == 2:
                roc_auc = roc_auc_score(
                    ref_bool, preds_bool
                )

            # gives the index of the top k predictions for each sample
            log_dict = {
                "eval_loss": eval_loss,
                "ndcg": ndcg_score(user_rating_preds, user_rating_ref),
                "novelty": novelty,
                "prediction_coverage": prediction_coverage,
                "catalog_coverage": catalog_coverage,
                "personalization": personalization,
                "roc_auc": roc_auc,
            }

            print(log_dict)
            if self.use_wandb:
                wandb.log(log_dict)


def main(
    use_wandb: bool = False,
    num_epochs: int = 100,
    eval_every: int = 1,
    max_batches: int = 100,
):
    print("Loading dataset..")
    dataset = MovieLens20MDataset(
        "ml-25m", Params.rating_format, Params.max_rows, Params.max_users
    )
    train_size = len(dataset) - Params.eval_size
    no_users, no_movies = dataset.no_movies, dataset.no_users
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, Params.eval_size]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=Params.batch_size, shuffle=False, drop_last=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=Params.eval_size, shuffle=False
    )
    if Params.model_architecture == ModelArchitecture.MATRIX_FACTORIZATION:
        model = MatrixFactorizationModel(no_movies, no_users, Params.embedding_dim)
    elif Params.model_architecture == ModelArchitecture.NEURAL_CF:
        model = NeuralCFModel(
            no_movies,
            no_users,
            Params.layers,
            Params.dropout,
            Params.rating_format,
        )
    elif Params.model_architecture == ModelArchitecture.DEEP_FM:
        model = DeepFMModel(
            no_movies, no_users, layers=Params.layers, dropout=Params.dropout
        )
    elif Params.model_architecture == ModelArchitecture.WIDE_DEEP:
        model = WideDeepModel(
            no_movies,
            no_users,
            Params.embedding_dim,
            Params.layers,
        )
    model.train()
    module = RecommenderModule(model, use_wandb)
    if use_wandb:
        wandb.init(project="recsys", config=vars(Params))
        wandb.watch(model)
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=Params.learning_rate, weight_decay=Params.weight_decay
    )
    for i in range(num_epochs):
        if i % eval_every == 0:
            print("Running eval..")
            for j, batch in enumerate(eval_dataloader):
                module.eval_step(dataset, batch, 10)
                break
        for j, batch in enumerate(train_dataloader):
            loss = module.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            grads = [
                param.grad.detach().flatten()
                for param in module.parameters()
                if param.grad is not None
            ]
            total_norm = torch.cat(grads).norm()
            if use_wandb:
                wandb.log({"total_norm": total_norm.item()})

            print(
                f"Epoch {i:03.0f}, batch {j:03.0f}, loss {loss.item():03.3f}, total norm: {total_norm.item():03.3f}"
            )

            if j > max_batches:
                break

            torch.nn.utils.clip_grad_norm_(module.parameters(), 100)


if __name__ == "__main__":
    fire.Fire(main)
