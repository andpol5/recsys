from enum import IntEnum
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from dataset import RatingFormat


class ModelArchitecture(IntEnum):
    NEURAL_CF = 1
    MATRIX_FACTORIZATION = 2
    DEEP_FM = 3
    WIDE_DEEP = 4


class RecModel(nn.Module):
    def __init__(
        self,
        emb_columns: List[str],
        feature_sizes: List[int],
        embedding_dim: int,
        rating_format: RatingFormat,
    ):
        super().__init__()

        self.rating_format = rating_format

        emb_dict = {}
        self.emb_columns = emb_columns
        for i, col_name in enumerate(emb_columns):
            emb_dict[col_name] = nn.Embedding(feature_sizes[i], embedding_dim)
        emb_dict = nn.ModuleDict(emb_dict)
        self.emb_dict = emb_dict
        self.emb_in_size = embedding_dim * len(emb_columns)

    def get_feature_embeddings(self, batch, concat=True):
        features, _ = batch
        embeddings = []
        for i, feature_name in enumerate(self.emb_columns):
            emb = self.emb_dict[feature_name]
            feature_column = features[:, i].to(torch.int64)
            embedded_column = emb(feature_column)
            embeddings.append(embedded_column)
        embeddings = torch.stack(embeddings, dim=1).squeeze()
        if concat:
            embeddings = embeddings.view(-1, self.emb_in_size)
        return embeddings


class WideDeepModel(RecModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.wide_layer = torch.nn.Linear(self.emb_in_size, 1)
        layers = [64, 32, 16, 8]

        self.deep_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.deep_layers.append(torch.nn.Linear(in_size, out_size))
            self.deep_layers.append(torch.nn.ReLU())
            self.deep_layers.append(torch.nn.Dropout(0.1))
        self.deep_layers = torch.nn.Sequential(*self.deep_layers)
        self.deep_output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, batch):
        emb_cat = self.get_feature_embeddings(batch)
        wide_out = self.wide_layer(emb_cat)
        deep_out = self.deep_layers(emb_cat)
        deep_out = self.deep_output_layer(deep_out)
        x = wide_out + deep_out
        if self.rating_format == RatingFormat.BINARY:
            x = torch.sigmoid(x)
        return x


class DeepFMModel(RecModel):
    def __init__(self) -> None:
        super().__init__()
        self.fc_layers = torch.nn.ModuleList()
        layers = [64, 32, 16, 8]

        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(torch.nn.ReLU())
            self.fc_layers.append(torch.nn.Dropout(0.1))

        self.fc_layers = torch.nn.Sequential(*self.fc_layers)
        self.output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, batch):
        emb_cat = self.get_feature_embeddings(batch)
        x = self.fc_layers(x)
        x = x + get_fm_loss(emb_cat)
        x = self.output_layer(x)
        if self.rating_format == RatingFormat.BINARY:
            x = torch.sigmoid(x)
        return x


def get_fm_loss(emb_cat: torch.Tensor):
    square_of_sum = torch.pow(torch.sum(emb_cat, dim=1, keepdim=True), 2)
    sum_of_square = torch.sum(emb_cat * emb_cat, dim=1, keepdim=True)
    cross_term = square_of_sum - sum_of_square
    cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
    return cross_term


class MatrixFactorizationModel(RecModel):
    def forward(self, batch):
        embeddings = self.get_feature_embeddings(batch, concat=False)
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(0)
        embeddings_prod = torch.prod(embeddings, dim=1)
        interaction = torch.sum(embeddings_prod, dim=1)
        return interaction


class NeuralCFModel(RecModel):
    def __init__(
        self,
        emb_columns: List[str],
        feature_sizes: List[str],
        embedding_dim: int,
        rating_format: RatingFormat,
        layers: List[int] = [64, 32, 16, 8],
    ):
        super().__init__(emb_columns, feature_sizes, embedding_dim, rating_format)
        assert layers[0] % 2 == 0, "layers[0] must be an even number"

        self.rating_format = rating_format

        self.bn = nn.BatchNorm1d(layers[0])

        self.fc_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # 1 is the output dimension
        self.output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, batch):
        x = self.get_feature_embeddings(batch)
        x = self.bn(x)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        rating = self.output_layer(x)
        if self.rating_format == RatingFormat.BINARY:
            rating = torch.sigmoid(rating)
        return rating


models_dict = {
    ModelArchitecture.MATRIX_FACTORIZATION: MatrixFactorizationModel,
    ModelArchitecture.NEURAL_CF: NeuralCFModel,
    ModelArchitecture.DEEP_FM: DeepFMModel,
    ModelArchitecture.WIDE_DEEP: WideDeepModel,
}
