import gc
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Set
from pickle import load

import click
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import TFLEX.expression
from TFLEX.assistance.toolbox.data.dataloader import SingledirectionalOneShotIterator
from TFLEX.assistance.toolbox.evaluate.GatherMetric import AverageMeter
from TFLEX.assistance.toolbox.exp.Experiment import Experiment
from TFLEX.assistance.toolbox.exp.OutputSchema import OutputSchema
from TFLEX.assistance.toolbox.utils.Progbar import Progbar
from TFLEX.assistance.toolbox.utils.RandomSeeds import set_seeds

from ComplexNumericalQueryDataloader import TestDataset, TrainDataset  #, TestDataset2
from NTFLEX_DSL import NeuralParser, is_to_predict_entity_set, is_to_predict_timestamp_set, query_contains_union_and_we_should_use_DNF, contains_relation
from ComplexNumericalQueryData import WIKI, ComplexNumericalQueryDatasetCachePath, NumericalComplexQueryData, TYPE_train_queries_answers, groups
from ParamSchemaExtension import is_entity, is_relation, is_timestamp, is_attribute, is_value

import os

#os.environ["TORCH_USE_CUDA_DSA"] = "1"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

QueryStructure = str
TYPE_token = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

L = 1


def convert_to_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def convert_to_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * L
    return y


def convert_to_time_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * L
    return y


def convert_to_time_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def convert_to_value_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * L
    return y


def convert_to_value_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


class EntityProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(EntityProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        token_dim = dim * 6
        self.layer1 = nn.Linear(token_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, token_dim)
        for i in range(2, num_layers + 1):
            setattr(self, f"layer{i}", nn.Linear(self.hidden_dim, self.hidden_dim))
        for i in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, f"layer{i}").weight)

    def forward(self,
                q_feature, q_logic, q_time_feature, q_time_logic, q_value_feature, q_value_logic,
                r_feature, r_logic, r_time_feature, r_time_logic, r_value_feature, r_value_logic,
                t_feature, t_logic, t_time_feature, t_time_logic, t_value_feature, t_value_logic):
        x = torch.cat([
            q_feature + r_feature + t_feature,
            q_logic + r_logic + t_logic,
            q_time_feature + r_time_feature + t_time_feature,
            q_time_logic + r_time_logic + t_time_logic,
            q_value_feature + r_value_feature + t_value_feature,
            q_value_logic + r_value_logic + t_value_logic, 
        ], dim=-1)
        for i in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, f"layer{i}")(x))
        x = self.layer0(x)

        feature, logic, time_feature, time_logic, value_feature, value_logic = torch.chunk(x, 6, dim=-1)
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        time_feature = convert_to_time_feature(time_feature)
        time_logic = convert_to_time_logic(time_logic)
        value_feature = convert_to_value_feature(value_feature)
        value_logic = convert_to_value_logic(value_logic)
        return feature, logic, time_feature, time_logic, value_feature, value_logic


class TimeProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(TimeProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        token_dim = dim * 6
        self.layer1 = nn.Linear(token_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, token_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self,
                q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_value_feature, q1_value_logic,
                r_feature, r_logic, r_time_feature, r_time_logic, r_value_feature, r_value_logic,
                q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_value_feature, q2_value_logic):
        x = torch.cat([
            q1_feature + r_feature + q2_feature,
            q1_logic + r_logic + q2_logic,
            q1_time_feature + r_time_feature + q2_time_feature,
            q1_time_logic + r_time_logic + q2_time_logic,
            q1_value_feature + r_value_feature + q2_value_feature,
            q1_value_logic + r_value_logic + q2_value_logic,
        ], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        feature, logic, time_feature, time_logic, value_feature, value_logic = torch.chunk(x, 6, dim=-1)
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        time_feature = convert_to_time_feature(time_feature)
        time_logic = convert_to_time_logic(time_logic)
        value_feature = convert_to_value_feature(value_feature)
        value_logic = convert_to_value_logic(value_logic)
        return feature, logic, time_feature, time_logic, value_feature, value_logic
    

class AttributeValueProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(AttributeValueProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        token_dim = dim * 6
        self.layer1 = nn.Linear(token_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, token_dim)
        for i in range(2, num_layers + 1):
            setattr(self, f"layer{i}", nn.Linear(self.hidden_dim, self.hidden_dim))
        for i in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, f"layer{i}").weight)

    def forward(self,
                q_feature, q_logic, q_time_feature, q_time_logic, q_value_feature, q_value_logic,
                a_feature, a_logic, a_time_feature, a_time_logic, a_value_feature, a_value_logic,
                t_feature, t_logic, t_time_feature, t_time_logic, t_value_feature, t_value_logic):
        x = torch.cat([
            q_feature + a_feature + t_feature,
            q_logic + a_logic + t_logic,
            q_time_feature + a_time_feature + t_time_feature,
            q_time_logic + a_time_logic + t_time_logic,
            q_value_feature + a_value_feature + t_value_feature,
            q_value_logic + a_value_logic + t_value_logic,
        ], dim=-1)
        for i in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, f"layer{i}")(x))
        x = self.layer0(x)

        feature, logic, time_feature, time_logic, value_feature, value_logic = torch.chunk(x, 6, dim=-1)
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        time_feature = convert_to_time_feature(time_feature)
        time_logic = convert_to_time_logic(time_logic)
        value_feature = convert_to_value_feature(value_feature)
        value_logic = convert_to_value_logic(value_logic)
        return feature, logic, time_feature, time_logic, value_feature, value_logic
    

class AttributeTimeProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(AttributeTimeProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        token_dim = dim * 6
        self.layer1 = nn.Linear(token_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, token_dim)
        for i in range(2, num_layers + 1):
            setattr(self, f"layer{i}", nn.Linear(self.hidden_dim, self.hidden_dim))
        for i in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, f"layer{i}").weight)

    def forward(self,
                q_feature, q_logic, q_time_feature, q_time_logic, q_value_feature, q_value_logic,
                a_feature, a_logic, a_time_feature, a_time_logic, a_value_feature, a_value_logic,
                x_feature, x_logic, x_time_feature, x_time_logic, x_value_feature, x_value_logic):
        x = torch.cat([
            q_feature + a_feature + x_feature,
            q_logic + a_logic + x_logic,
            q_time_feature + a_time_feature + x_time_feature,
            q_time_logic + a_time_logic + x_time_logic,
            q_value_feature + a_value_feature + x_value_feature,
            q_value_logic + a_value_logic + x_value_logic,
        ], dim=-1)
        for i in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, f"layer{i}")(x))
        x = self.layer0(x)

        feature, logic, time_feature, time_logic, value_feature, value_logic = torch.chunk(x, 6, dim=-1)
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        time_feature = convert_to_time_feature(time_feature)
        time_logic = convert_to_time_logic(time_logic)
        value_feature = convert_to_value_feature(value_feature)
        value_logic = convert_to_value_logic(value_logic)
        return feature, logic, time_feature, time_logic, value_feature, value_logic
    

class AttributeReverseProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(AttributeReverseProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        token_dim = dim * 6
        self.layer1 = nn.Linear(token_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, token_dim)
        for i in range(2, num_layers + 1):
            setattr(self, f"layer{i}", nn.Linear(self.hidden_dim, self.hidden_dim))
        for i in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, f"layer{i}").weight)

    def forward(self,
                x_feature, x_logic, x_time_feature, x_time_logic, x_value_feature, x_value_logic,
                a_feature, a_logic, a_time_feature, a_time_logic, a_value_feature, a_value_logic,
                t_feature, t_logic, t_time_feature, t_time_logic, t_value_feature, t_value_logic):
        x = torch.cat([
            x_feature + a_feature + t_feature,
            x_logic + a_logic + t_logic,
            x_time_feature + a_time_feature + t_time_feature,
            x_time_logic + a_time_logic + t_time_logic,
            x_value_feature + a_value_feature + t_value_feature,
            x_value_logic + a_value_logic + t_value_logic,
        ], dim=-1)
        for i in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, f"layer{i}")(x))
        x = self.layer0(x)

        feature, logic, time_feature, time_logic, value_feature, value_logic = torch.chunk(x, 6, dim=-1)
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        time_feature = convert_to_time_feature(time_feature)
        time_logic = convert_to_time_logic(time_logic)
        value_feature = convert_to_value_feature(value_feature)
        value_logic = convert_to_value_logic(value_logic)
        return feature, logic, time_feature, time_logic, value_feature, value_logic


class EntityIntersection(nn.Module):
    def __init__(self, dim):
        super(EntityIntersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.value_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.value_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logits = torch.cat([value_feature, value_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.value_feature_layer_2(F.relu(self.value_feature_layer_1(logits))), dim=0)
        value_feature = torch.sum(feature_attention * value_feature, dim=0)

        logic, _ = torch.min(logic, dim=0)
        time_logic, _ = torch.min(time_logic, dim=0)
        value_logic, _ = torch.min(value_logic, dim=0)
        return feature, logic, time_feature, time_logic, value_feature, value_logic


class TemporalIntersection(nn.Module):
    def __init__(self, dim):
        super(TemporalIntersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.value_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.value_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logits = torch.cat([value_feature, value_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.value_feature_layer_2(F.relu(self.value_feature_layer_1(logits))), dim=0)
        value_feature = torch.sum(feature_attention * value_feature, dim=0)

        logic, _ = torch.min(logic, dim=0)
        time_logic, _ = torch.min(time_logic, dim=0)
        value_logic, _ = torch.min(value_logic, dim=0)
        return feature, logic, time_feature, time_logic, value_feature, value_logic
    

class ValueIntersection(nn.Module):
    def __init__(self, dim):
        super(ValueIntersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.value_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.value_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logits = torch.cat([value_feature, value_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.value_feature_layer_2(F.relu(self.value_feature_layer_1(logits))), dim=0)
        value_feature = torch.sum(feature_attention * value_feature, dim=0)

        logic, _ = torch.min(logic, dim=0)
        time_logic, _ = torch.min(time_logic, dim=0)
        value_logic, _ = torch.min(value_logic, dim=0)
        return feature, logic, time_feature, time_logic, value_feature, value_logic
    

class EntityUnion(nn.Module):
    def __init__(self, dim):
        super(EntityUnion, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.value_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.value_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logits = torch.cat([value_feature, value_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.value_feature_layer_2(F.relu(self.value_feature_layer_1(logits))), dim=0)
        value_feature = torch.sum(feature_attention * value_feature, dim=0)

        logic, _ = torch.max(logic, dim=0)
        # for time, it is intersection
        time_logic, _ = torch.min(time_logic, dim=0)
        # logic = torch.prod(logic, dim=0)
        value_logic, _ = torch.min(value_logic, dim=0)
        return feature, logic, time_feature, time_logic, value_feature, value_logic


class TemporalUnion(nn.Module):
    def __init__(self, dim):
        super(TemporalUnion, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.value_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.value_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logits = torch.cat([value_feature, value_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.value_feature_layer_2(F.relu(self.value_feature_layer_1(logits))), dim=0)
        value_feature = torch.sum(feature_attention * value_feature, dim=0)

        # for entity, it is intersection
        logic, _ = torch.min(logic, dim=0)
        # for time, it is union
        time_logic, _ = torch.max(time_logic, dim=0)
        # logic = torch.prod(logic, dim=0)
        value_logic, _ = torch.min(value_logic, dim=0)
        return feature, logic, time_feature, time_logic, value_feature, value_logic
    

class ValueUnion(nn.Module):
    def __init__(self, dim):
        super(ValueUnion, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.value_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.value_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.value_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logits = torch.cat([value_feature, value_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.value_feature_layer_2(F.relu(self.value_feature_layer_1(logits))), dim=0)
        value_feature = torch.sum(feature_attention * value_feature, dim=0)

        # for entity, it is intersection
        logic, _ = torch.min(logic, dim=0)
        # for time, it is union
        time_logic, _ = torch.min(time_logic, dim=0)
        # logic = torch.prod(logic, dim=0)
        value_logic, _ = torch.max(value_logic, dim=0)
        return feature, logic, time_feature, time_logic, value_feature, value_logic


class EntityNegation(nn.Module):
    def __init__(self, dim):
        super(EntityNegation, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature = self.feature_layer_2(F.relu(self.feature_layer_1(logits)))
        logic = 1 - logic
        return feature, logic, time_feature, time_logic, value_feature, value_logic


class TemporalNegation(nn.Module):
    def __init__(self, dim):
        super(TemporalNegation, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        time_feature = self.feature_layer_2(F.relu(self.feature_layer_1(logits)))
        time_logic = 1 - time_logic
        return feature, logic, time_feature, time_logic, value_feature, value_logic
    

class ValueNegation(nn.Module):
    def __init__(self, dim):
        super(ValueNegation, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        logits = torch.cat([value_feature, value_logic], dim=-1)  # N x B x 2d
        value_feature = self.feature_layer_2(F.relu(self.feature_layer_1(logits)))
        value_logic = 1 - value_logic
        return feature, logic, time_feature, time_logic, value_feature, value_logic


def scale_feature(feature):
    # f,f' in [-L, L]
    # f' = (f + 2L) % (2L) - L, where L=1
    indicator_positive = feature >= 0
    indicator_negative = feature < 0
    feature[indicator_positive] = feature[indicator_positive] - L
    feature[indicator_negative] = feature[indicator_negative] + L
    return feature


class TemporalBefore(nn.Module):
    def __init__(self, dim):
        super(TemporalBefore, self).__init__()
        self.dim = dim

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        time_feature = scale_feature(time_feature - L / 2 - time_logic / 2)
        time_logic = (L - time_logic) / 2

        return feature, logic, time_feature, time_logic, value_feature, value_logic


class TemporalAfter(nn.Module):
    def __init__(self, dim):
        super(TemporalAfter, self).__init__()
        self.dim = dim

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        time_feature = scale_feature(time_feature + L / 2 + time_logic / 2)
        time_logic = (L - time_logic) / 2

        return feature, logic, time_feature, time_logic, value_feature, value_logic


class TemporalNext(nn.Module):
    def __init__(self):
        super(TemporalNext, self).__init__()

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        time_feature = scale_feature(time_feature)
        time_logic = 1 - time_logic
        return feature, logic, time_feature, time_logic, value_feature, value_logic
    

class ValueGreater(nn.Module):
    def __init__(self, dim):
        super(ValueGreater, self).__init__()
        self.dim = dim

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        value_feature = scale_feature(value_feature + L / 2 + value_logic / 2)
        value_logic = (L - value_logic) / 2

        return feature, logic, time_feature, time_logic, value_feature, value_logic


class ValueSmaller(nn.Module):
    def __init__(self, dim):
        super(ValueSmaller, self).__init__()
        self.dim = dim

    def forward(self, feature, logic, time_feature, time_logic, value_feature, value_logic):
        value_feature = scale_feature(value_feature - L / 2 - value_logic / 2)
        value_logic = (L - value_logic) / 2

        return feature, logic, time_feature, time_logic, value_feature, value_logic


class NTFLEX(nn.Module):
    def __init__(self,
                 nentity, nrelation, nattribute, nvalue, ntimestamp, hidden_dim, gamma,
                 test_batch_size=1, center_reg=None, drop: float = 0.
                 ):
        super(NTFLEX, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.nattribute = nattribute
        self.nvalue = nvalue
        self.ntimestamp = ntimestamp
        self.attribute_dim = hidden_dim
        self.value_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.timestamp_dim = hidden_dim

        # entity only have feature part but no logic part
        self.entity_feature_embedding = nn.Embedding(nentity, self.entity_dim)

        self.timestamp_feature_embedding = nn.Embedding(ntimestamp, self.timestamp_dim)

        self.value_feature_embedding = nn.Embedding(nvalue, self.value_dim)

        self.relation_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_logic_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_time_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_time_logic_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_value_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_value_logic_embedding = nn.Embedding(nrelation, self.relation_dim)

        self.attribute_feature_embedding = nn.Embedding(nattribute, self.attribute_dim)
        self.attribute_logic_embedding = nn.Embedding(nattribute, self.attribute_dim)
        self.attribute_time_feature_embedding = nn.Embedding(nattribute, self.attribute_dim)
        self.attribute_time_logic_embedding = nn.Embedding(nattribute, self.attribute_dim)
        self.attribute_value_feature_embedding = nn.Embedding(nattribute, self.attribute_dim)
        self.attribute_value_logic_embedding = nn.Embedding(nattribute, self.attribute_dim)

        self.entity_projection = EntityProjection(hidden_dim, drop=drop)
        self.entity_intersection = EntityIntersection(hidden_dim)
        self.entity_union = EntityUnion(hidden_dim)
        self.entity_negation = EntityNegation(hidden_dim)

        self.time_projection = TimeProjection(hidden_dim, drop=drop)
        self.time_intersection = TemporalIntersection(hidden_dim)
        self.time_union = TemporalUnion(hidden_dim)
        self.time_negation = TemporalNegation(hidden_dim)
        self.time_before = TemporalBefore(hidden_dim)
        self.time_after = TemporalAfter(hidden_dim)
        self.time_next = TemporalNext()

        self.attribute_value_projection = AttributeValueProjection(hidden_dim, drop=drop)
        self.attribute_time_projection = AttributeTimeProjection(hidden_dim, drop=drop)
        self.attribute_reverse_projection = AttributeReverseProjection(hidden_dim, drop=drop)
        self.value_intersection = ValueIntersection(hidden_dim)
        self.value_union = ValueUnion(hidden_dim)
        self.value_negation = ValueNegation(hidden_dim)
        self.value_greater = ValueGreater(hidden_dim)
        self.value_smaller = ValueSmaller(hidden_dim)

        self.batch_entity_range = torch.arange(nentity).float().repeat(test_batch_size, 1)
        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False)
        embedding_range = self.embedding_range.item()
        self.modulus = nn.Parameter(torch.Tensor([0.5 * embedding_range]), requires_grad=True)
        self.cen = center_reg
        self.parser = self.build_parser()

    def build_neural_ops(self):
        def And(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_value_feature, q1_value_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_value_feature, q2_value_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            value_feature = torch.stack([q1_value_feature, q2_value_feature])
            value_logic = torch.stack([q1_value_logic, q2_value_logic])
            return self.entity_intersection(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def And3(q1, q2, q3):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_value_feature, q1_value_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_value_feature, q2_value_logic = q2
            q3_feature, q3_logic, q3_time_feature, q3_time_logic, q3_value_feature, q3_value_logic = q3
            feature = torch.stack([q1_feature, q2_feature, q3_feature])
            logic = torch.stack([q1_logic, q2_logic, q3_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature, q3_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic, q3_time_logic])
            value_feature = torch.stack([q1_value_feature, q2_value_feature, q3_value_feature])
            value_logic = torch.stack([q1_value_logic, q2_value_logic, q3_value_logic])
            return self.entity_intersection(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def Or(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_value_feature, q1_value_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_value_feature, q2_value_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            value_feature = torch.stack([q1_value_feature, q2_value_feature])
            value_logic = torch.stack([q1_value_logic, q2_value_logic])
            return self.entity_union(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def Not(q):
            feature, logic, time_feature, time_logic, value_feature, value_logic = q
            return self.entity_negation(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def TimeNot(q):
            feature, logic, time_feature, time_logic, value_feature, value_logic = q
            return self.time_negation(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def EntityProjection2(e1, r1, t1):
            s_feature, s_logic, s_time_feature, s_time_logic, s_value_feature, s_value_logic = e1
            r_feature, r_logic, r_time_feature, r_time_logic, r_value_feature, r_value_logic = r1
            t_feature, t_logic, t_time_feature, t_time_logic, t_value_feature, t_value_logic = t1
            return self.entity_projection(
                s_feature, s_logic, s_time_feature, s_time_logic, s_value_feature, s_value_logic,
                r_feature, r_logic, r_time_feature, r_time_logic, r_value_feature, r_value_logic,
                t_feature, t_logic, t_time_feature, t_time_logic, t_value_feature, t_value_logic
            )

        def TimeProjection2(e1, r1, e2):
            s_feature, s_logic, s_time_feature, s_time_logic, s_value_feature, s_value_logic = e1
            r_feature, r_logic, r_time_feature, r_time_logic, r_value_feature, r_value_logic = r1
            o_feature, o_logic, o_time_feature, o_time_logic, o_value_feature, o_value_logic = e2
            return self.time_projection(
                s_feature, s_logic, s_time_feature, s_time_logic, s_value_feature, s_value_logic,
                r_feature, r_logic, r_time_feature, r_time_logic, r_value_feature, r_value_logic,
                o_feature, o_logic, o_time_feature, o_time_logic, o_value_feature, o_value_logic
            )
        
        def AttributeValueProjection2(e1, a1, t1):
            s_feature, s_logic, s_time_feature, s_time_logic, s_value_feature, s_value_logic = e1
            a_feature, a_logic, a_time_feature, a_time_logic, a_value_feature, a_value_logic = a1
            t_feature, t_logic, t_time_feature, t_time_logic, t_value_feature, t_value_logic = t1
            return self.attribute_value_projection(
                s_feature, s_logic, s_time_feature, s_time_logic, s_value_feature, s_value_logic,
                a_feature, a_logic, a_time_feature, a_time_logic, a_value_feature, a_value_logic,
                t_feature, t_logic, t_time_feature, t_time_logic, t_value_feature, t_value_logic
            )
        
        def AttributeTimeProjection2(e1, a1, x1):
            s_feature, s_logic, s_time_feature, s_time_logic, s_value_feature, s_value_logic = e1
            a_feature, a_logic, a_time_feature, a_time_logic, a_value_feature, a_value_logic = a1
            x_feature, x_logic, x_time_feature, x_time_logic, x_value_feature, x_value_logic = x1
            return self.attribute_time_projection(
                s_feature, s_logic, s_time_feature, s_time_logic, s_value_feature, s_value_logic,
                a_feature, a_logic, a_time_feature, a_time_logic, a_value_feature, a_value_logic,
                x_feature, x_logic, x_time_feature, x_time_logic, x_value_feature, x_value_logic
            )
        
        def AttributeReverseProjection2(x1, a1, t1):
            x_feature, x_logic, x_time_feature, x_time_logic, x_value_feature, x_value_logic = x1
            a_feature, a_logic, a_time_feature, a_time_logic, a_value_feature, a_value_logic = a1
            t_feature, t_logic, t_time_feature, t_time_logic, t_value_feature, t_value_logic = t1
            return self.attribute_reverse_projection(
                x_feature, x_logic, x_time_feature, x_time_logic, x_value_feature, x_value_logic,
                a_feature, a_logic, a_time_feature, a_time_logic, a_value_feature, a_value_logic,
                t_feature, t_logic, t_time_feature, t_time_logic, t_value_feature, t_value_logic
            )

        def TimeAnd(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_value_feature, q1_value_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_value_feature, q2_value_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            value_feature = torch.stack([q1_value_feature, q2_value_feature])
            value_logic = torch.stack([q1_value_logic, q2_value_logic])
            return self.time_intersection(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def TimeAnd3(q1, q2, q3):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_value_feature, q1_value_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_value_feature, q2_value_logic = q2
            q3_feature, q3_logic, q3_time_feature, q3_time_logic, q3_value_feature, q3_value_logic = q3
            feature = torch.stack([q1_feature, q2_feature, q3_feature])
            logic = torch.stack([q1_logic, q2_logic, q3_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature, q3_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic, q3_time_logic])
            value_feature = torch.stack([q1_value_feature, q2_value_feature, q3_value_feature])
            value_logic = torch.stack([q1_value_logic, q2_value_logic, q3_value_logic])
            return self.time_intersection(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def TimeOr(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_value_feature, q1_value_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_value_feature, q2_value_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            value_feature = torch.stack([q1_value_feature, q2_value_feature])
            value_logic = torch.stack([q1_value_logic, q2_value_logic])
            return self.time_union(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def TimeBefore(q):
            feature, logic, time_feature, time_logic, value_feature, value_logic = q
            return self.time_before(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def TimeAfter(q):
            feature, logic, time_feature, time_logic, value_feature, value_logic = q
            return self.time_after(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def TimeNext(q):
            feature, logic, time_feature, time_logic, value_feature, value_logic = q
            return self.time_next(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def beforePRt(e1, r1, e2):
            return TimeBefore(TimeProjection2(e1, r1, e2))

        def afterPRt(e1, r1, e2):
            return TimeAfter(TimeProjection2(e1, r1, e2))

        def beforePAt(e1, a1, x1):
            return TimeBefore(AttributeTimeProjection2(e1, a1, x1))

        def afterPAt(e1, a1, x1):# -> Any:
            return TimeAfter(AttributeTimeProjection2(e1, a1, x1))
        
        def ValueAnd(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_value_feature, q1_value_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_value_feature, q2_value_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            value_feature = torch.stack([q1_value_feature, q2_value_feature])
            value_logic = torch.stack([q1_value_logic, q2_value_logic])
            return self.value_intersection(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def ValueOr(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic, q1_value_feature, q1_value_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic, q2_value_feature, q2_value_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            value_feature = torch.stack([q1_value_feature, q2_value_feature])
            value_logic = torch.stack([q1_value_logic, q2_value_logic])
            return self.value_union(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def ValueGreater2(q, a):
            feature, logic, time_feature, time_logic, value_feature, value_logic = q
            return self.value_greater(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def ValueSmaller2(q, a):
            feature, logic, time_feature, time_logic, value_feature, value_logic = q
            return self.value_smaller(feature, logic, time_feature, time_logic, value_feature, value_logic)

        def greaterPAx(e1, a1, t1):
            return ValueGreater2(AttributeValueProjection2(e1, a1, t1), a1)

        def smallerPAx(e1, a1, t1):
            return ValueSmaller2(AttributeValueProjection2(e1, a1, t1), a1)

        neural_ops = {
            "And": And,
            "And3": And3,
            "Or": Or,
            "AttributeValueProjection": AttributeValueProjection2,
            "AttributeReverseProjection": AttributeReverseProjection2,
            "AttributeTimeProjection": AttributeTimeProjection2,
            "EntityProjection": EntityProjection2,
            "TimeProjection": TimeProjection2,
            "ValueAnd": ValueAnd,
            "ValueOr": ValueOr,
            "ValueSmaller": ValueSmaller2,
            "ValueGreater": ValueGreater2,
            "TimeAnd": TimeAnd,
            "TimeAnd3": TimeAnd3,
            "TimeOr": TimeOr,
            "TimeBefore": TimeBefore,
            "TimeAfter": TimeAfter,
            "TimeNext": TimeNext,
            "afterPRt": afterPRt,
            "beforePRt": beforePRt,
            "afterPAt": afterPAt,
            "beforePAt": beforePAt,
            "greaterPAx": greaterPAx,
            "smallerPAx": smallerPAx,
        }
        return neural_ops

    def build_parser(self):
        neural_ops = self.build_neural_ops()
        return NeuralParser(neural_ops)

    def init(self):
        embedding_range = self.embedding_range.item()
        nn.init.uniform_(tensor=self.entity_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)

        nn.init.uniform_(tensor=self.timestamp_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)

        nn.init.uniform_(tensor=self.value_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)

        nn.init.uniform_(tensor=self.relation_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_time_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_time_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_value_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.relation_value_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)

        nn.init.uniform_(tensor=self.attribute_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.attribute_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.attribute_time_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.attribute_time_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.attribute_value_feature_embedding.weight.data, a=-embedding_range, b=embedding_range)
        nn.init.uniform_(tensor=self.attribute_value_logic_embedding.weight.data, a=-embedding_range, b=embedding_range)

    def scale(self, embedding):
        return embedding / self.embedding_range

    def entity_feature(self, idx):
        return convert_to_feature(self.scale(self.entity_feature_embedding(idx)))

    def timestamp_feature(self, idx):
        return convert_to_time_feature(self.scale(self.timestamp_feature_embedding(idx)))
    
    def value_feature(self, idx):
        return convert_to_value_feature(self.scale(self.value_feature_embedding(idx)))

    def entity_token(self, idx) -> TYPE_token:
        feature = self.entity_feature(idx)
        logic = torch.zeros_like(feature).to(feature.device)
        time_feature = torch.zeros_like(feature).to(feature.device)
        time_logic = torch.zeros_like(feature).to(feature.device)
        value_feature = torch.zeros_like(feature).to(feature.device)
        value_logic = torch.zeros_like(feature).to(feature.device)
        # print("entity", f"{feature.max().item()}, {feature.min().item()}",
        #       f"{logic.max().item()}, {logic.min().item()}",
        #       f"{time_feature.max().item()}, {time_feature.min().item()}",
        #       f"{time_logic.max().item()}, {time_logic.min().item()}",
        #       f"{value_feature.max().item()}, {value_feature.min().item()}",
        #       f"{value_logic.max().item()}, {value_logic.min().item()}"
        #       )
        # print("entity", f"{idx.max().item()}, {idx.min().item()}")
        return feature, logic, time_feature, time_logic, value_feature, value_logic

    def timestamp_token(self, idx) -> TYPE_token:
        time_feature = self.timestamp_feature(idx)
        feature = torch.zeros_like(time_feature).to(time_feature.device)
        logic = torch.zeros_like(time_feature).to(time_feature.device)
        time_logic = torch.zeros_like(time_feature).to(time_feature.device)
        value_feature = torch.zeros_like(time_feature).to(time_feature.device)
        value_logic = torch.zeros_like(time_feature).to(time_feature.device)
        # print("timestamp", f"{feature.max().item()}, {feature.min().item()}",
        #       f"{logic.max().item()}, {logic.min().item()}",
        #       f"{time_feature.max().item()}, {time_feature.min().item()}",
        #       f"{time_logic.max().item()}, {time_logic.min().item()}",
        #       f"{value_feature.max().item()}, {value_feature.min().item()}",
        #       f"{value_logic.max().item()}, {value_logic.min().item()}"
        #       )
        # print("timestamp", f"{idx.max().item()}, {idx.min().item()}")
        return feature, logic, time_feature, time_logic, value_feature, value_logic
    
    def value_token(self, idx) -> TYPE_token:
        value_feature = self.value_feature(idx)
        feature = torch.zeros_like(value_feature).to(value_feature.device)
        logic = torch.zeros_like(value_feature).to(value_feature.device)
        time_feature = torch.zeros_like(value_feature).to(value_feature.device)
        time_logic = torch.zeros_like(value_feature).to(value_feature.device)
        value_logic = torch.zeros_like(value_feature).to(value_feature.device)
        # print("value", f"{feature.max().item()}, {feature.min().item()}",
        #       f"{logic.max().item()}, {logic.min().item()}",
        #       f"{time_feature.max().item()}, {time_feature.min().item()}",
        #       f"{time_logic.max().item()}, {time_logic.min().item()}",
        #       f"{value_feature.max().item()}, {value_feature.min().item()}",
        #       f"{value_logic.max().item()}, {value_logic.min().item()}"
        #       )
        # print("value", f"{idx.max().item()}, {idx.min().item()}")
        return feature, logic, time_feature, time_logic, value_feature, value_logic
    
    def relation_token(self, idx) -> TYPE_token:
        feature = convert_to_feature(self.scale(self.relation_feature_embedding(idx)))
        logic = convert_to_logic(self.scale(self.relation_logic_embedding(idx)))
        time_feature = convert_to_time_feature(self.scale(self.relation_time_feature_embedding(idx)))
        time_logic = convert_to_time_logic(self.scale(self.relation_time_logic_embedding(idx)))
        value_feature = convert_to_value_feature(self.scale(self.relation_value_feature_embedding(idx)))
        value_logic = convert_to_value_logic(self.scale(self.relation_value_logic_embedding(idx)))
        # print("relation", f"{feature.max().item()}, {feature.min().item()}",
        #       f"{logic.max().item()}, {logic.min().item()}",
        #       f"{time_feature.max().item()}, {time_feature.min().item()}",
        #       f"{time_logic.max().item()}, {time_logic.min().item()}",
        #       f"{value_feature.max().item()}, {value_feature.min().item()}",
        #       f"{value_logic.max().item()}, {value_logic.min().item()}"
        #       )
        # print("relation", f"{idx.max().item()}, {idx.min().item()}")
        return feature, logic, time_feature, time_logic, value_feature, value_logic
       
    def attribute_token(self, idx) -> TYPE_token:
        feature = convert_to_feature(self.scale(self.attribute_feature_embedding(idx)))
        logic = convert_to_logic(self.scale(self.attribute_logic_embedding(idx)))
        time_feature = convert_to_time_feature(self.scale(self.attribute_time_feature_embedding(idx)))
        time_logic = convert_to_time_logic(self.scale(self.attribute_time_logic_embedding(idx)))
        value_feature = convert_to_value_feature(self.scale(self.attribute_value_feature_embedding(idx)))
        value_logic = convert_to_value_logic(self.scale(self.attribute_value_logic_embedding(idx)))
        # print("attribute", f"{feature.max().item()}, {feature.min().item()}",
        #       f"{logic.max().item()}, {logic.min().item()}",
        #       f"{time_feature.max().item()}, {time_feature.min().item()}",
        #       f"{time_logic.max().item()}, {time_logic.min().item()}",
        #       f"{value_feature.max().item()}, {value_feature.min().item()}",
        #       f"{value_logic.max().item()}, {value_logic.min().item()}"
        #       )
        # print("attribute", f"{idx.max().item()}, {idx.min().item()}")
        return feature, logic, time_feature, time_logic, value_feature, value_logic

    def embed_args(self, query_args: List[str], query_tensor: torch.Tensor) -> TYPE_token:
        embedding_of_args = []
        for i in range(len(query_args)):
            arg_name = query_args[i]
            tensor = query_tensor[:, i]
            if is_entity(arg_name):
                token_embedding = self.entity_token(tensor)
            elif is_relation(arg_name):
                token_embedding = self.relation_token(tensor)
            elif is_timestamp(arg_name):
                token_embedding = self.timestamp_token(tensor)
            elif is_value(arg_name):
                token_embedding = self.value_token(tensor)
            elif is_attribute(arg_name):
                token_embedding = self.attribute_token(tensor)
            else:
                raise Exception("Unknown Args %s" % arg_name)
            embedding_of_args.append(token_embedding)
        return tuple(embedding_of_args)

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        return self.forward_FLEX(
            positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def forward_FLEX(self,
                     positive_answer: Optional[torch.Tensor],
                     negative_answer: Optional[torch.Tensor],
                     subsampling_weight: Optional[torch.Tensor],
                     grouped_query: Dict[QueryStructure, torch.Tensor],
                     grouped_idxs: Dict[QueryStructure, List[List[int]]]):
        """
        positive_answer: None or (B, )
        negative_answer: None or (B, N)
        subsampling_weight: None or (B, )
        """
        (all_idxs_e, all_predict_e), \
            (all_idxs_t, all_predict_t), \
            (all_idxs_v, all_predict_v), \
            (all_union_idxs_e, all_union_predict_e), \
            (all_union_idxs_t, all_union_predict_t), \
            (all_union_idxs_v, all_union_predict_v) = self.batch_predict(grouped_query, grouped_idxs)

        all_idxs = all_idxs_e + all_idxs_t + all_idxs_v + all_union_idxs_e + all_union_idxs_t + all_union_idxs_v
        if subsampling_weight is not None:
            subsampling_weight = subsampling_weight[all_idxs]

        positive_scores = None
        negative_scores = None

        # 2. 计算正例损失
        if positive_answer is not None:
            scores_e = self.scoring_to_answers_by_idxs(
                all_idxs_e, positive_answer, all_predict_e, predict_entity=1, DNF_predict=False)
            scores_t = self.scoring_to_answers_by_idxs(
                all_idxs_t, positive_answer, all_predict_t, predict_entity=2, DNF_predict=False)
            scores_v = self.scoring_to_answers_by_idxs(
                all_idxs_v, positive_answer, all_predict_v, predict_entity=3, DNF_predict=False)
            scores_union_e = self.scoring_to_answers_by_idxs(
                all_union_idxs_e, positive_answer, all_union_predict_e, predict_entity=1, DNF_predict=True)
            scores_union_t = self.scoring_to_answers_by_idxs(
                all_union_idxs_t, positive_answer, all_union_predict_t, predict_entity=2, DNF_predict=True)
            scores_union_v = self.scoring_to_answers_by_idxs(
                all_union_idxs_v, positive_answer, all_union_predict_v, predict_entity=3, DNF_predict=True)
            positive_scores = torch.cat([scores_e, scores_t, scores_v, scores_union_e, scores_union_t, scores_union_v], dim=0)

        # 3. 计算负例损失
        if negative_answer is not None:
            scores_e = self.scoring_to_answers_by_idxs(
                all_idxs_e, negative_answer, all_predict_e, predict_entity=1, DNF_predict=False)
            scores_t = self.scoring_to_answers_by_idxs(
                all_idxs_t, negative_answer, all_predict_t, predict_entity=2, DNF_predict=False)
            scores_v = self.scoring_to_answers_by_idxs(
                all_idxs_v, negative_answer, all_predict_v, predict_entity=3, DNF_predict=False)
            scores_union_e = self.scoring_to_answers_by_idxs(
                all_union_idxs_e, negative_answer, all_union_predict_e, predict_entity=1, DNF_predict=True)
            scores_union_t = self.scoring_to_answers_by_idxs(
                all_union_idxs_t, negative_answer, all_union_predict_t, predict_entity=2, DNF_predict=True)
            scores_union_v = self.scoring_to_answers_by_idxs(
                all_union_idxs_v, negative_answer, all_union_predict_v, predict_entity=3, DNF_predict=True)
            negative_scores = torch.cat([scores_e, scores_t, scores_v, scores_union_e, scores_union_t, scores_union_v], dim=0)

        return positive_scores, negative_scores, subsampling_weight, all_idxs

    def single_predict(self, query_structure: QueryStructure, query_tensor: torch.Tensor) -> Union[TYPE_token,
                                                                                                   Tuple[TYPE_token, TYPE_token]]:
        query_name, query_args = query_structure
        # other query and DM are normal
        func = self.parser.fast_function(query_name)
        embedding_of_args = self.embed_args(query_args, query_tensor)  # [B x dt]*L
        predict = func(*embedding_of_args)  # B x dt
        return predict

    def batch_predict(self, grouped_query: Dict[QueryStructure, torch.Tensor],
                      grouped_idxs: Dict[QueryStructure, List[List[int]]]):
        all_idxs_e, all_predict_e = [], []
        all_idxs_t, all_predict_t = [], []
        all_idxs_v, all_predict_v = [], []
        all_union_idxs_e, all_union_predict_1_e, all_union_predict_2_e = [], [], []
        all_union_idxs_t, all_union_predict_1_t, all_union_predict_2_t = [], [], []
        all_union_idxs_v, all_union_predict_1_v, all_union_predict_2_v = [], [], []
        all_union_predict_e: Optional[TYPE_token] = None
        all_union_predict_t: Optional[TYPE_token] = None
        all_union_predict_v: Optional[TYPE_token] = None

        for query_structure in grouped_query:
            query_name = query_structure
            query_args = self.parser.fast_args(query_name)
            query_tensor = grouped_query[query_structure]  # (B, L), B for batch size, L for query args length
            query_idxs = grouped_idxs[query_structure]
            # query_idxs is of shape Bx1.
            # each element indicates global index of each row in query_tensor.
            # global index means the index in sample from dataloader.
            # the sample is grouped by query name and leads to query_tensor here.
            # other query and DM are normal
            func = self.parser.fast_function(query_name)
            embedding_of_args = self.embed_args(query_args, query_tensor)  # (B, d)*L
            predict = func(*embedding_of_args)  # (B, d)
            if is_to_predict_entity_set(query_name):
                all_predict_e.append(predict)
                all_idxs_e.extend(query_idxs)
            elif is_to_predict_timestamp_set(query_name):
                all_predict_t.append(predict)
                all_idxs_t.extend(query_idxs)
            else:
                all_predict_v.append(predict)
                all_idxs_v.extend(query_idxs)

        def cat_to_tensor(token_list: List[TYPE_token]) -> TYPE_token:
            feature = []
            logic = []
            time_feature = []
            time_logic = []
            value_feature = []
            value_logic = []
            for x in token_list:
                feature.append(x[0])
                logic.append(x[1])
                time_feature.append(x[2])
                time_logic.append(x[3])
                value_feature.append(x[4])
                value_logic.append(x[5])
            feature = torch.cat(feature, dim=0).unsqueeze(1)
            logic = torch.cat(logic, dim=0).unsqueeze(1)
            time_feature = torch.cat(time_feature, dim=0).unsqueeze(1)
            time_logic = torch.cat(time_logic, dim=0).unsqueeze(1)
            value_feature = torch.cat(value_feature, dim=0).unsqueeze(1)
            value_logic = torch.cat(value_logic, dim=0).unsqueeze(1)
            return feature, logic, time_feature, time_logic, value_feature, value_logic

        if len(all_idxs_e) > 0:
            all_predict_e = cat_to_tensor(all_predict_e)  # (B, 1, d) * 5
        if len(all_idxs_t) > 0:
            all_predict_t = cat_to_tensor(all_predict_t)  # (B, 1, d) * 5
        if len(all_idxs_v) > 0:
            all_predict_v = cat_to_tensor(all_predict_v)
        if len(all_union_idxs_e) > 0:
            all_union_predict_1_e = cat_to_tensor(all_union_predict_1_e)  # (B, 1, d) * 5
            all_union_predict_2_e = cat_to_tensor(all_union_predict_2_e)  # (B, 1, d) * 5
            all_union_predict_e: TYPE_token = tuple([torch.cat([x, y], dim=1) for x, y in zip(
                all_union_predict_1_e, all_union_predict_2_e)])  # (B, 2, d) * 5
        if len(all_union_idxs_t) > 0:
            all_union_predict_1_t = cat_to_tensor(all_union_predict_1_t)  # (B, 1, d) * 5
            all_union_predict_2_t = cat_to_tensor(all_union_predict_2_t)  # (B, 1, d) * 5
            all_union_predict_t: TYPE_token = tuple([torch.cat([x, y], dim=1) for x, y in zip(
                all_union_predict_1_t, all_union_predict_2_t)])  # (B, 2, d) * 5
        if len(all_union_idxs_v) > 0:
            all_union_predict_1_v = cat_to_tensor(all_union_predict_1_v)  # (B, 1, d) * 5
            all_union_predict_2_v = cat_to_tensor(all_union_predict_2_v)  # (B, 1, d) * 5
            all_union_predict_v: TYPE_token = tuple([torch.cat([x, y], dim=1) for x, y in zip(
                all_union_predict_1_v, all_union_predict_2_v)])  # (B, 2, d) * 5
        return (all_idxs_e, all_predict_e), \
               (all_idxs_t, all_predict_t), \
               (all_idxs_v, all_predict_v), \
               (all_union_idxs_e, all_union_predict_e), \
               (all_union_idxs_t, all_union_predict_t), \
               (all_union_idxs_v, all_union_predict_v)

    def grouped_predict(self, grouped_query: Dict[QueryStructure, torch.Tensor],
                        grouped_answer: Dict[QueryStructure, torch.Tensor]) -> Dict[QueryStructure, torch.Tensor]:
        grouped_score = {}
        for query_structure in grouped_query:
            query_batch = grouped_query[query_structure]  # (B, L), B for batch size, L for query args length
            answer = grouped_answer[query_structure]  # (B, N)
            grouped_score[query_structure] = self.forward_predict(query_structure, query_batch, answer)

        return grouped_score
    
    def grouped_predict_rel(self, grouped_query: Dict[QueryStructure, torch.Tensor],
                        grouped_answer: Dict[QueryStructure, torch.Tensor]) -> Dict[QueryStructure, torch.Tensor]:
        grouped_score = {}
        for query_structure in grouped_query:
            for rel in grouped_query[query_structure]:
                query_batch = grouped_query[query_structure][rel]  # (B, L), B for batch size, L for query args length
                answer = grouped_answer[query_structure][rel]  # (B, N)
                grouped_score[query_structure] = {rel: self.forward_predict(query_structure, query_batch, answer)}

        return grouped_score

    def forward_predict(
            self, query_structure: QueryStructure, query_tensor: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        # query_tensor  # (B, L), B for batch size, L for query args length
        # answer  # (B, N)
        query_name = query_structure
        query_args = self.parser.fast_args(query_name)
        # the sample is grouped by query name and leads to query_tensor here.
        func = self.parser.fast_function(query_name)
        embedding_of_args = self.embed_args(query_args, query_tensor)  # (B, d)*L
        predict = func(*embedding_of_args)  # (B, d)
        all_predict: TYPE_token = tuple([i.unsqueeze(dim=1) for i in predict])  # (B, 1, d)
        if is_to_predict_entity_set(query_name):
            return self.scoring_to_answers(answer, all_predict, predict_entity=1, DNF_predict=False)
        elif is_to_predict_timestamp_set(query_name):
            return self.scoring_to_answers(answer, all_predict, predict_entity=2, DNF_predict=False)
        else:
            return self.scoring_to_answers(answer, all_predict, predict_entity=3, DNF_predict=False)

    def scoring_to_answers_by_idxs(
            self, all_idxs, answer: torch.Tensor, q: TYPE_token, predict_entity=1, DNF_predict=False):
        """
        B for batch size
        N for negative sampling size (maybe N=1 when positive samples only)
        all_answer_idxs: (B, ) or (B, N) int
        all_predict:     (B, 1, dt) or (B, 2, dt) float
        return score:    (B, N) float
        """
        if len(all_idxs) <= 0:
            return torch.Tensor([]).to(self.embedding_range.device)
        answer_ids = answer[all_idxs]
        answer_ids = answer_ids.view(answer_ids.shape[0], -1)
        return self.scoring_to_answers(answer_ids, q, predict_entity, DNF_predict)

    def scoring_to_answers(self, answer_ids: torch.Tensor, q: TYPE_token, predict_entity=1, DNF_predict=False):
        """
        B for batch size
        N for negative sampling size (maybe N=1 when positive samples only)
        answer_ids:   (B, N) int
        all_predict:  (B, 1, dt) or (B, 2, dt) float
        return score: (B, N) float
        """
        q: TYPE_token = tuple([i.unsqueeze(dim=2) for i in q])  # (B, 1, 1, dt) or (B, 2, 1, dt)
        if predict_entity == 1:
            feature = self.entity_feature(answer_ids).unsqueeze(dim=1)  # (B, 1, N, d)
            scores = self.scoring_entity(feature, q)  # (B, 1, N) or (B, 2, N)
        elif predict_entity == 2:
            feature = self.timestamp_feature(answer_ids).unsqueeze(dim=1)  # (B, 1, N, d)
            scores = self.scoring_timestamp(feature, q)  # (B, 1, N) or (B, 2, N)
        else:
            feature = self.value_feature(answer_ids).unsqueeze(dim=1)  # (B, 1, N, d)
            scores = self.scoring_value(feature, q)  # (B, 1, N) or (B, 2, N)

        if DNF_predict:
            scores = torch.max(scores, dim=1)[0]  # (B, N)
        else:
            scores = scores.squeeze(dim=1)  # (B, N)
        return scores  # (B, N)

    def distance_between_entity_and_query(self, entity_feature, query_feature, query_logic):
        """
        entity_feature (B, 1, N, d)
        query_feature  (B, 1, 1, dt) or (B, 2, 1, dt)
        query_logic    (B, 1, 1, dt) or (B, 2, 1, dt)
        query    =                 [(feature - logic) | feature | (feature + logic)]
        entity   = entity_feature            |             |               |
                         |                   |             |               |
        1) from entity to center of the interval           |               |
        d_center = entity_feature -                     feature            |
                         |<------------------------------->|               |
        2) from entity to left of the interval                             |
        d_left   = entity_feature - (feature - logic)                      |
                         |<----------------->|                             |
        3) from entity to right of the interval                            |
        d_right  = entity_feature -                               (feature + logic)
                         |<----------------------------------------------->|
        """
        d_center = entity_feature - query_feature
        d_left = entity_feature - (query_feature - query_logic)
        d_right = entity_feature - (query_feature + query_logic)

        # inner distance
        feature_distance = torch.abs(d_center)
        inner_distance = torch.min(feature_distance, query_logic)
        # outer distance
        outer_distance = torch.min(torch.abs(d_left), torch.abs(d_right))
        outer_distance[feature_distance < query_logic] = 0.  # if entity is inside, we don't care about outer.

        distance = torch.norm(outer_distance, p=1, dim=-1) + self.cen * torch.norm(inner_distance, p=1, dim=-1)
        return distance

    def distance_between_timestamp_and_query(self, timestamp_feature, time_feature, time_logic):
        """
        entity_feature (B, 1, N, d)
        query_feature  (B, 1, 1, dt) or (B, 2, 1, dt)
        query_logic    (B, 1, 1, dt) or (B, 2, 1, dt)
        query    =                 [(feature - logic) | feature | (feature + logic)]
        entity   = entity_feature            |             |               |
                         |                   |             |               |
        1) from entity to center of the interval           |               |
        d_center = entity_feature -                     feature            |
                         |<------------------------------->|               |
        2) from entity to left of the interval                             |
        d_left   = entity_feature - (feature - logic)                      |
                         |<----------------->|                             |
        3) from entity to right of the interval                            |
        d_right  = entity_feature -                               (feature + logic)
                         |<----------------------------------------------->|
        """
        d_center = timestamp_feature - time_feature
        d_left = timestamp_feature - (time_feature - time_logic)
        d_right = timestamp_feature - (time_feature + time_logic)

        # inner distance
        feature_distance = torch.abs(d_center)
        inner_distance = torch.min(feature_distance, time_logic)
        # outer distance
        outer_distance = torch.min(torch.abs(d_left), torch.abs(d_right))
        outer_distance[feature_distance < time_logic] = 0.  # if entity is inside, we don't care about outer.

        distance = torch.norm(outer_distance, p=1, dim=-1) + self.cen * torch.norm(inner_distance, p=1, dim=-1)
        return distance
    
    def distance_between_value_and_query(self, attribute_feature, value_feature, value_logic):
        """
        entity_feature (B, 1, N, d)
        query_feature  (B, 1, 1, dt) or (B, 2, 1, dt)
        query_logic    (B, 1, 1, dt) or (B, 2, 1, dt)
        query    =                 [(feature - logic) | feature | (feature + logic)]
        entity   = entity_feature            |             |               |
                         |                   |             |               |
        1) from entity to center of the interval           |               |
        d_center = entity_feature -                     feature            |
                         |<------------------------------->|               |
        2) from entity to left of the interval                             |
        d_left   = entity_feature - (feature - logic)                      |
                         |<----------------->|                             |
        3) from entity to right of the interval                            |
        d_right  = entity_feature -                               (feature + logic)
                         |<----------------------------------------------->|
        """
        d_center = attribute_feature - value_feature
        d_left = attribute_feature - (value_feature - value_logic)
        d_right = attribute_feature - (value_feature + value_logic)

        # inner distance
        feature_distance = torch.abs(d_center)
        inner_distance = torch.min(feature_distance, value_logic)
        # outer distance
        outer_distance = torch.min(torch.abs(d_left), torch.abs(d_right))
        outer_distance[feature_distance < value_logic] = 0.  # if entity is inside, we don't care about outer.

        distance = torch.norm(outer_distance, p=1, dim=-1) + self.cen * torch.norm(inner_distance, p=1, dim=-1)
        return distance

    def scoring_entity(self, entity_feature, q: TYPE_token):
        feature, logic, time_feature, time_logic, value_feature, value_logic = q
        distance = self.distance_between_entity_and_query(entity_feature, feature, logic)
        score = self.gamma - distance * self.modulus
        return score

    def scoring_timestamp(self, timestamp_feature, q: TYPE_token):
        feature, logic, time_feature, time_logic, value_feature, value_logic = q
        distance = self.distance_between_timestamp_and_query(timestamp_feature, time_feature, time_logic)
        score = self.gamma - distance * self.modulus
        return score
    
    def scoring_value(self, attribute_feature, q: TYPE_token):
        feature, logic, time_feature, time_logic, value_feature, value_logic = q
        distance = self.distance_between_value_and_query(attribute_feature, value_feature, value_logic)
        score = self.gamma - distance * self.modulus
        return score
    

class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema, data: NumericalComplexQueryData, model,
                 start_step, max_steps, every_test_step, every_valid_step,
                 batch_size, test_batch_size, negative_sample_size,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, cpu_num,
                 hidden_dim, input_dropout, gamma, center_reg, train_tasks, train_all, eval_tasks, eval_all
                 ):
        super(MyExperiment, self).__init__(output)
        self.log(f"{locals()}")

        self.model_param_store.save_scripts([__file__])
        entity_count = data.entity_count
        timestamp_count = data.timestamp_count
        value_count = data.value_count
        self.groups = groups
        self.log('-------------------------------' * 3)
        self.log('# entity: %d' % data.entity_count)
        self.log('# attribute: %d' % data.attribute_count)
        self.log('# relation: %d' % data.relation_count)
        self.log('# timestamp: %d' % data.timestamp_count)
        self.log('# value: %d' % data.value_count)
        self.log('# max steps: %d' % max_steps)

        # 1. build train dataset
        if train_all:
            data.load_cache(["train_queries_answers"])
        else:
            data.train_queries_answers = data.load_cache_by_tasks(train_tasks.split(","), "train")

        if eval_all:
            tasks = []
            for group in self.groups:
                if group != "relations":
                    tasks.extend(self.groups[group])
        else:
            tasks = eval_tasks.split(",")
        data.valid_queries_answers = data.load_cache_by_tasks(tasks, "valid")
        data.test_queries_answers = data.load_cache_by_tasks(tasks, "test")

        train_queries_answers = data.train_queries_answers
        valid_queries_answers = data.valid_queries_answers
        test_queries_answers = data.test_queries_answers

        train_path_queries: TYPE_train_queries_answers = {}
        train_other_queries: TYPE_train_queries_answers = {}
        path_list = ["PRe", "PRt", "PAe", "PAt", "PAx"]
        for query_structure_name in train_queries_answers:
            if query_structure_name in path_list:
                train_path_queries[query_structure_name] = train_queries_answers[query_structure_name]
            else:
                train_other_queries[query_structure_name] = train_queries_answers[query_structure_name]

        self.log("Training info:")
        for query_structure_name in train_queries_answers:
            self.log(query_structure_name + ": " +
                     str(len(train_queries_answers[query_structure_name]["queries_answers"])))
        del train_queries_answers
        del data.train_queries_answers
        gc.collect()

        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(train_path_queries, entity_count, value_count, timestamp_count, negative_sample_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))
        if len(train_other_queries) > 0:
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(train_other_queries, entity_count, value_count, timestamp_count, negative_sample_size),
                batch_size=batch_size,
                shuffle=True,
                num_workers=cpu_num,
                collate_fn=TrainDataset.collate_fn
            ))
        else:
            train_other_iterator = None

        del train_path_queries
        del train_other_queries
        gc.collect()

        valid_dataloader = DataLoader(
            TestDataset(valid_queries_answers, entity_count, value_count, timestamp_count),
            batch_size=test_batch_size,
            num_workers=max(cpu_num // 2, 1),
            collate_fn=TestDataset.collate_fn_rel
        )

        test_dataloader = DataLoader(
            TestDataset(test_queries_answers, entity_count, value_count, timestamp_count),
            batch_size=test_batch_size,
            num_workers=max(1, cpu_num // 2),
            collate_fn=TestDataset.collate_fn_rel
        )

        self.log("Validation info:")
        for query_structure_name in valid_queries_answers:
            self.log(query_structure_name + ": " +
                     str(len(valid_queries_answers[query_structure_name]["queries_answers"])))
        self.log("Test info:")
        for query_structure_name in test_queries_answers:
            self.log(query_structure_name + ": " +
                     str(len(test_queries_answers[query_structure_name]["queries_answers"])))

        del valid_queries_answers
        del test_queries_answers
        del data.valid_queries_answers
        del data.test_queries_answers
        gc.collect()

        data.load_cache(["idx2attribute", "idx2relation"])

        self.idx2relation = data.idx2relation
        self.idx2attribute = data.idx2attribute
        del data.idx2relation
        del data.idx2attribute
        gc.collect()

        # 2. build model
        model = model.to(train_device)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        best_score = 0
        best_test_score = 0
        train_loss_dict = {}
        if resume:
            if resume_by_score > 0:
                start_step, _, best_score = self.model_param_store.load_by_score(model, opt, resume_by_score)
            else:
                start_step, _, best_score = self.model_param_store.load_best(model, opt)
            self.dump_model(model)
            model.eval()
            with torch.no_grad():
                self.debug("Resumed from score %.4f." % best_score)
                self.debug("Take a look at the performance after resumed.")
                self.debug("Test (step: %d):" % start_step)
                result = self.evaluate_with_rel(model, test_dataloader, test_device)
                best_test_score, _ = self.visual_result(start_step + 1, result, "Test")
                self.debug("Validation (step: %d):" % start_step)
                result = self.evaluate_with_rel(model, valid_dataloader, test_device)
                best_score, _ = self.visual_result(start_step + 1, result, "Valid")
        else:
            model.init()
            self.dump_model(model)

        current_learning_rate = lr
        hyper = {
            'center_reg': center_reg,
            'learning_rate': lr,
            'batch_size': batch_size,
            "hidden_dim": hidden_dim,
            "gamma": gamma,
        }
        self.metric_log_store.add_hyper(hyper)
        for k, v in hyper.items():
            self.log(f'{k} = {v}')
        self.metric_log_store.add_progress(max_steps)
        warm_up_steps = max_steps // 2

        # 3. training
        progbar = Progbar(max_step=max_steps)
        for step in range(start_step, max_steps):
            model.train()
            log = self.train(model, opt, train_path_iterator, step, train_device)
            for metric in log:
                self.visualize_store.add_scalar('path_' + metric, log[metric], step)
            if train_other_iterator is not None:
                log = self.train(model, opt, train_other_iterator, step, train_device)
                for metric in log:
                    self.visualize_store.add_scalar('other_' + metric, log[metric], step)
                log = self.train(model, opt, train_path_iterator, step, train_device)

            train_loss_dict[step] = log["loss"]
            progbar.update(
                step + 1,
                [("step", step + 1),
                 ("loss", log["loss"]),
                 ("positive", log["positive_sample_loss"]),
                 ("negative", log["negative_sample_loss"])])
            if (step + 1) % 10 == 0:
                self.metric_log_store.add_loss(log, step + 1)

            if (step + 1) >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                print("")
                self.log('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                opt = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5

            if (step + 1) % every_valid_step == 0:
                model.eval()
                with torch.no_grad():
                    print("")
                    self.debug("Validation (step: %d):" % (step + 1))
                    result = self.evaluate_with_rel(model, valid_dataloader, test_device)
                    score, row_results = self.visual_result(step + 1, result, "Valid")
                    if score >= best_score:
                        self.success("current score=%.4f > best score=%.4f" % (score, best_score))
                        best_score = score
                        self.metric_log_store.add_best_metric({"result": result}, "Valid")
                        self.debug("saving best score %.4f" % score)
                        self.model_param_store.save_best(model, opt, step, 0, score)
                        self.latex_store.save_best_valid_result(row_results)
                    else:
                        self.model_param_store.save_by_score(model, opt, step, 0, score)
                        self.latex_store.save_valid_result_by_score(row_results, score)
                        self.fail("current score=%.4f < best score=%.4f" % (score, best_score))
            if (step + 1) % every_test_step == 0:
                model.eval()
                with torch.no_grad():
                    print("")
                    self.debug("Test (step: %d):" % (step + 1))
                    result = self.evaluate_with_rel(model, test_dataloader, test_device)
                    score, row_results = self.visual_result(step + 1, result, "Test")
                    self.latex_store.save_test_result_by_score(row_results, score)
                    if score >= best_test_score:
                        best_test_score = score
                        self.latex_store.save_best_test_result(row_results)
                        self.metric_log_store.add_best_metric({"result": result}, "Test")
                    print("")

        # 5. report the best
        start_step, _, best_score = self.model_param_store.load_best(model, opt)
        model.eval()
        with torch.no_grad():
            self.debug("Reporting the best performance...")
            self.debug("Resumed from score %.4f." % best_score)
            self.debug("Take a look at the performance after resumed.")
            self.debug("Test (step: %d):" % start_step)
            result = self.evaluate_with_rel(model, test_dataloader, test_device)
            best_test_score, _ = self.visual_result(start_step + 1, result, "Test")
            self.debug("Validation (step: %d):" % start_step)
            result = self.evaluate_with_rel(model, valid_dataloader, test_device)
            best_score, _ = self.visual_result(start_step + 1, result, "Valid")
        self.metric_log_store.finish()
        
        # Retrieve each dictionary's values
        train_values = train_loss_dict.values()
        train_keys = train_loss_dict.keys()
        # Generate a sequence of integers to represent the epoch numbers
        # Plot and label the training and validation loss values
        plt.plot(train_keys, train_values, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig("trainig_plot.pdf") 

    def train(self, model, optimizer, train_iterator, step, device="cuda:0"):
        model.train()
        model.to(device)
        optimizer.zero_grad()

        grouped_query, grouped_idxs, positive_answer, negative_answer, subsampling_weight = next(train_iterator)
        for key in grouped_query:
            grouped_query[key] = grouped_query[key].to(device)
        positive_answer = positive_answer.to(device)
        negative_answer = negative_answer.to(device)
        subsampling_weight = subsampling_weight.to(device)

        positive_logit, negative_logit, subsampling_weight, _ = model(
            positive_answer, negative_answer, subsampling_weight, grouped_query, grouped_idxs)

        negative_sample_loss = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_sample_loss = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_sample_loss).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_sample_loss).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log    

    def evaluate(self, model, test_dataloader, device="cuda:0"):
        model.to(device)
        total_steps = len(test_dataloader)
        progbar = Progbar(max_step=total_steps)
        logs = defaultdict(lambda: {
            'filtered-MRR': AverageMeter('filtered-MRR'),
            'MSRR': AverageMeter('MSRR'),
            'MRR': AverageMeter('MRR'),
            'hits@1': AverageMeter('hits@1'),
            'hits@3': AverageMeter('hits@3'),
            'hits@10': AverageMeter('hits@10'),
            'num_queries': AverageMeter('num_queries'),
            'num_answers': AverageMeter('num_answers'),
        })
        step = 0
        for grouped_query, grouped_candidate_answer, grouped_easy_answer, grouped_hard_answer_dict, grouped_hard_answer_mask in test_dataloader:
            for query_name in grouped_query:
                grouped_query[query_name] = grouped_query[query_name].to(device)            
                grouped_candidate_answer[query_name] = grouped_candidate_answer[query_name].to(device)

            grouped_score = model.grouped_predict(grouped_query, grouped_candidate_answer)
            for query_name in grouped_score:            
                score = grouped_score[query_name]
                easy_answer_mask: List[torch.Tensor] = grouped_easy_answer[query_name]
                hard_answer_mask: List[torch.Tensor] = grouped_hard_answer_mask[query_name]
                hard_answer: List[Set[int]] = grouped_hard_answer_dict[query_name]
                candidate_answers: torch.Tensor = grouped_candidate_answer[query_name]
                # we remove easy answer, because easy answer may exist in training set
                score[easy_answer_mask] = -float('inf')
                hard_scores = score[hard_answer_mask]
                score[hard_answer_mask] = -float('inf')
                negative_ranking, _ = torch.sort(score, dim=1, descending=True)  # sorted entity idx (B, N)
                positive_ranking, _ = torch.sort(hard_scores, descending=True)
                positive_ranking = torch.unsqueeze(positive_ranking, 1)
                num_queries = score.shape[0]
                logs[query_name]['num_queries'].update(num_queries)
                for i in range(num_queries):
                    negative_rank_i = negative_ranking[i]
                    positive_rank_i = positive_ranking[i]
                    rank_of_answers = torch.searchsorted(-negative_rank_i, -positive_rank_i)
                    
                    hard_answer_i = torch.tensor(list(hard_answer[i]), device=device)
                    candidate_answer_i = candidate_answers[i]

                    # MSRR, mean set reciprocal rank
                    # MSR, mean set rank
                    num_of_easy_answer = len(easy_answer_mask[i].nonzero())
                    num_of_candidates, num_of_answers = len(candidate_answer_i), len(hard_answer_i)
                    expect_ranks = candidate_answer_i[:num_of_answers]
                    MSRR = torch.mean(1 - (rank_of_answers-expect_ranks) / (num_of_candidates-num_of_easy_answer-num_of_answers)).item()
                    logs[query_name]['MSRR'].update(MSRR)

                    logs[query_name]['num_answers'].update(len(rank_of_answers))

                    # mrr, hits@k
                    for rank in rank_of_answers.cpu().numpy().data:
                        logs[query_name]['MRR'].update(1 / (rank + 1))
                        logs[query_name]['hits@1'].update(1.0 if rank < 1 else 0.0)
                        logs[query_name]['hits@3'].update(1.0 if rank < 3 else 0.0)
                        logs[query_name]['hits@10'].update(1.0 if rank < 10 else 0.0)

                    del hard_answer_i
            step += 1
            progbar.update(step, [("Hits @10", logs[query_name]['hits@10'].avg), ("query", ",".join(list(grouped_query.keys())))])

        metrics = defaultdict(lambda: defaultdict(int))
        for query_name in logs:
            for metric in logs[query_name]:
                if metric == "num_queries":
                    metrics[query_name][metric] = logs[query_name][metric].sum
                else:
                    metrics[query_name][metric] = logs[query_name][metric].avg

        return metrics
    

    def evaluate_max(self, model, test_dataloader, device="cuda:0"):
        model.to(device)
        total_steps = len(test_dataloader)
        progbar = Progbar(max_step=total_steps)
        logs = defaultdict(lambda: {
            'filtered-MRR': AverageMeter('filtered-MRR'),
            'MSRR': AverageMeter('MSRR'),
            'MRR': AverageMeter('MRR'),
            'hits@1': AverageMeter('hits@1'),
            'hits@3': AverageMeter('hits@3'),
            'hits@10': AverageMeter('hits@10'),
            'num_queries': AverageMeter('num_queries'),
            'num_answers': AverageMeter('num_answers'),
        })
        step = 0
        for grouped_query, grouped_candidate_answer, grouped_easy_answer, grouped_hard_answer_dict, _ in test_dataloader:
            for query_name in grouped_query:
                grouped_query[query_name] = grouped_query[query_name].to(device)            
                grouped_candidate_answer[query_name] = grouped_candidate_answer[query_name].to(device)

            grouped_score = model.grouped_predict(grouped_query, grouped_candidate_answer)
            for query_name in grouped_score:            
                score = grouped_score[query_name]
                easy_answer_mask: List[torch.Tensor] = grouped_easy_answer[query_name]
                hard_answer: List[Set[int]] = grouped_hard_answer_dict[query_name]
                candidate_answers: torch.Tensor = grouped_candidate_answer[query_name]
                # we remove easy answer, because easy answer may exist in training set
                score[easy_answer_mask] = -float('inf')
                ranking = score.argsort(dim=1, descending=True)  # sorted entity idx (B, N)
                num_queries = ranking.shape[0]
                logs[query_name]['num_queries'].update(num_queries)
                for i in range(num_queries):
                    rank_i = ranking[i]
                    hard_answer_i = torch.tensor(list(hard_answer[i]), device=device)
                    candidate_answer_i = candidate_answers[i]

                    rank_of_answers = torch.nonzero(rank_i.view(-1)[..., None] == hard_answer_i)[:, 0]

                    # MSRR, mean set reciprocal rank
                    # MSR, mean set rank
                    num_of_easy_answer = len(easy_answer_mask[i].nonzero())
                    num_of_candidates, num_of_answers = len(candidate_answer_i), len(hard_answer_i)
                    expect_ranks = candidate_answer_i[:num_of_answers]
                    MSRR = torch.mean(1 - (rank_of_answers-expect_ranks) / (num_of_candidates-num_of_easy_answer-num_of_answers)).item()
                    logs[query_name]['MSRR'].update(MSRR)

                    logs[query_name]['num_answers'].update(len(rank_of_answers))

                    # mrr, hits@k
                    rank = min(rank_of_answers.cpu().numpy().data)
                    logs[query_name]['MRR'].update(1 / (rank + 1))
                    logs[query_name]['hits@1'].update(1.0 if rank < 1 else 0.0)
                    logs[query_name]['hits@3'].update(1.0 if rank < 3 else 0.0)
                    logs[query_name]['hits@10'].update(1.0 if rank < 10 else 0.0)

                    del hard_answer_i
            step += 1
            progbar.update(step, [("Hits @10", logs[query_name]['hits@10'].avg), ("query", ",".join(list(grouped_query.keys())))])

        metrics = defaultdict(lambda: defaultdict(int))
        for query_name in logs:
            for metric in logs[query_name]:
                if metric == "num_queries":
                    metrics[query_name][metric] = logs[query_name][metric].sum
                else:
                    metrics[query_name][metric] = logs[query_name][metric].avg

        return metrics
    

    def evaluate_with_rel(self, model, test_dataloader, device="cuda:0"):
        model.to(device)
        total_steps = len(test_dataloader)
        progbar = Progbar(max_step=total_steps)
        logs = defaultdict(lambda: {
            'filtered-MRR': AverageMeter('filtered-MRR'),
            'MSRR': AverageMeter('MSRR'),
            'MRR': AverageMeter('MRR'),
            'hits@1': AverageMeter('hits@1'),
            'hits@3': AverageMeter('hits@3'),
            'hits@10': AverageMeter('hits@10'),
            'num_queries': AverageMeter('num_queries'),
            'num_answers': AverageMeter('num_answers'),
        })
        step = 0
        for grouped_query, grouped_candidate_answer, grouped_easy_answer, grouped_hard_answer_dict, grouped_hard_answer_mask in test_dataloader:
            for query_name in grouped_query:
                for rel in grouped_query[query_name]:
                    grouped_query[query_name][rel] = grouped_query[query_name][rel].to(device)            
                    grouped_candidate_answer[query_name][rel] = grouped_candidate_answer[query_name][rel].to(device)

            grouped_score = model.grouped_predict_rel(grouped_query, grouped_candidate_answer)
            for query_name in grouped_score:
                for rel in grouped_score[query_name]:
                    # PARSA: add try except to see how many queries failed
                    try:
                        if contains_relation(query_name):
                            if rel > model.nrelation // 2:
                                rel_name = self.idx2relation[rel - model.nrelation // 2 ]
                            else:
                                rel_name = self.idx2relation[rel]  
                        else:
                            if rel > model.nattribute // 2:
                                rel_name = self.idx2attribute[rel - model.nattribute // 2]
                            else:
                                rel_name = self.idx2attribute[rel]
                        
                        score = grouped_score[query_name][rel]
                        easy_answer_mask: List[torch.Tensor] = grouped_easy_answer[query_name][rel]
                        hard_answer_mask: List[torch.Tensor] = grouped_hard_answer_mask[query_name][rel]
                        hard_answer: List[Set[int]] = grouped_hard_answer_dict[query_name][rel]
                        candidate_answers: torch.Tensor = grouped_candidate_answer[query_name][rel]
                        # we remove easy answer, because easy answer may exist in training set
                        score[easy_answer_mask] = -float('inf')
                        hard_scores = score[hard_answer_mask]
                        score[hard_answer_mask] = -float('inf')
                        negative_ranking, _ = torch.sort(score, dim=1, descending=True)  # sorted entity idx (B, N)
                        positive_ranking, _ = torch.sort(hard_scores, descending=True)
                        positive_ranking = torch.unsqueeze(positive_ranking, 1)
                        num_queries = score.shape[0]
                        logs[query_name]['num_queries'].update(num_queries)
                        logs[rel_name]['num_queries'].update(num_queries)
                        for i in range(num_queries):
                            negative_rank_i = negative_ranking[i]
                            positive_rank_i = positive_ranking[i]
                            rank_of_answers = torch.searchsorted(-negative_rank_i, -positive_rank_i)
                            
                            hard_answer_i = torch.tensor(list(hard_answer[i]), device=device)
                            candidate_answer_i = candidate_answers[i]

                            # MSRR, mean set reciprocal rank
                            # MSR, mean set rank
                            num_of_easy_answer = len(easy_answer_mask[i].nonzero())
                            num_of_candidates, num_of_answers = len(candidate_answer_i), len(hard_answer_i)
                            expect_ranks = candidate_answer_i[:num_of_answers]
                            MSRR = torch.mean(1 - (rank_of_answers-expect_ranks) / (num_of_candidates-num_of_easy_answer-num_of_answers)).item()
                            logs[query_name]['MSRR'].update(MSRR)

                            logs[query_name]['num_answers'].update(len(rank_of_answers))

                            # mrr, hits@k
                            for rank in rank_of_answers.cpu().numpy().data:
                                logs[query_name]['MRR'].update(1 / (rank + 1))
                                logs[query_name]['hits@1'].update(1.0 if rank < 1 else 0.0)
                                logs[query_name]['hits@3'].update(1.0 if rank < 3 else 0.0)
                                logs[query_name]['hits@10'].update(1.0 if rank < 10 else 0.0)
                                logs[rel_name]['MRR'].update(1 / (rank + 1))
                                logs[rel_name]['hits@1'].update(1.0 if rank < 1 else 0.0)
                                logs[rel_name]['hits@3'].update(1.0 if rank < 3 else 0.0)
                                logs[rel_name]['hits@10'].update(1.0 if rank < 10 else 0.0)

                            del hard_answer_i
                    except Exception as e:
                        print("rel:", rel, "query_name:", query_name)
                        print(e)
            step += 1
            progbar.update(step, [("Hits @10", logs[query_name]['hits@10'].avg), ("query", ",".join(list(grouped_query.keys())))])

        metrics = defaultdict(lambda: defaultdict(int))
        for query_name in logs:
            for metric in logs[query_name]:
                if metric == "num_queries":
                    metrics[query_name][metric] = logs[query_name][metric].sum
                else:
                    metrics[query_name][metric] = logs[query_name][metric].avg

        return metrics
    
    def visual_result(self, step_num: int, result, scope: str):
        group_scores = {}
        group_row_results = defaultdict(list)
        for group, group_list in self.groups.items():
            group_result = {}
            for query_structure in group_list:
                if query_structure in result:
                    group_result[query_structure] = result[query_structure]
            score, row_results = self.visual_group_result(step_num, group_result, scope, group)
            group_scores[group] = score
            for row in row_results:
                group_row_results[row].extend(row_results[row])
        AVG = sum(group_scores.values()) / len(group_scores)
        self.log(f"AVG: {AVG:.2%}")
        return AVG, group_row_results

    def visual_group_result(self, step_num: int, result, scope: str, group: str):
        """Evaluate queries in dataloader"""
        self.metric_log_store.add_metric({scope+"_"+group: result}, step_num, scope+"_"+group)
        average_metrics = defaultdict(float)
        num_query_structures = 0
        num_queries = 0
        for query_structure in result:
            for metric in result[query_structure]:
                self.visualize_store.add_scalar(
                    "_".join([scope, group, query_structure, metric]),
                    result[query_structure][metric],
                    step_num)
                if metric != 'num_queries':
                    average_metrics[metric] += result[query_structure][metric]
            num_queries += result[query_structure]['num_queries']
            num_query_structures += 1

        for metric in average_metrics:
            average_metrics[metric] /= num_query_structures
            self.visualize_store.add_scalar(
                "_".join([scope, group, 'average', metric]),
                average_metrics[metric],
                step_num)

        header = "{0:<8s}".format(scope)
        row_results = defaultdict(list)
        row_results[header].append(group)
        row_results["num_queries"].append(num_queries)
        for row in average_metrics:
            cell = average_metrics[row]
            row_results[row].append(cell)
        for col in sorted(result):
            row_results[header].append(col)
            col_data = result[col]
            for row in col_data:
                cell = col_data[row]
                row_results[row].append(cell)

        def to_str(data):
            if isinstance(data, float):
                return "{0:>6.2%}  ".format(data)
            elif isinstance(data, int):
                return "{0:^6d}  ".format(data)
            else:
                return "{0:^6s}  ".format(data[:6])

        for i in row_results:
            row = row_results[i]
            self.log("{0:<8s}".format(i)[:8] + ": " + "".join([to_str(data) for data in row]))

        score = average_metrics["MRR"]
        return score, row_results


@click.command()
@click.option("--data_home", type=str, default="data", help="The folder path to dataset.")
@click.option("--dataset", type=str, default="WIKI", help="Which dataset to use: WIKI")
@click.option("--name", type=str, default="NTFLEX", help="Name of the experiment.")
@click.option("--start_step", type=int, default=0, help="start step.")
@click.option("--max_steps", type=int, default=100001, help="Number of steps.")
@click.option("--every_test_step", type=int, default=5000, help="test every k steps")
@click.option("--every_valid_step", type=int, default=5000, help="validation every k steps.")
@click.option("--batch_size", type=int, default=128, help="Batch size.")
@click.option("--test_batch_size", type=int, default=8,
              help="Test batch size. Scoring to all is memory consuming. We need small test batch size.")
@click.option('--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--resume", type=bool, default=False, help="Resume from output directory.")
@click.option("--resume_by_score", type=float, default=0.0,
              help="Resume by score from output directory. Resume best if it is 0. Default: 0")
@click.option("--lr", type=float, default=0.0001, help="Learning rate.")
@click.option('--cpu_num', type=int, default=1, help="used to speed up torch.dataloader")
@click.option('--hidden_dim', type=int, default=800, help="embedding dimension")
@click.option("--input_dropout", type=float, default=0.1, help="Input layer dropout.")
@click.option('--gamma', type=float, default=15.0, help="margin in the loss")
@click.option('--center_reg', type=float, default=0.02,
              help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
@click.option('--train_tasks', type=str, default="PAe,PRe,PRe2,ea2i,er2i,PAt,PRt,PRta,PRtb,ta2i,tr2i,PAx,gPAx,sPAx,gPAxi,sPAxi,gsPAxi", help='the tasks for training')
@click.option('--train_all', type=bool, default=False,
              help='if training all, it will use all tasks in data.train_queries_answers')
@click.option('--eval_tasks', type=str, default="PAe,PRe,PRe2,ea2i,er2i,PAt,PRt,PRta,PRtb,ta2i,tr2i,PAx,gPAx,sPAx,gPAxi,sPAxi,gsPAxi" + ",ea2u,er2u,ta2u,tr2u,gPAxu,sPAxu,gsPAxu", help='the tasks for evaluation')
@click.option('--eval_all', type=bool, default=False,
              help='if evaluating all, it will use all tasks in data.test_queries_answers')
def main(data_home, dataset, name,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, negative_sample_size,
         train_device, test_device,
         resume, resume_by_score,
         lr, cpu_num,
         hidden_dim, input_dropout, gamma, center_reg, train_tasks, train_all, eval_tasks, eval_all
         ):
    set_seeds(42)
    output = OutputSchema(dataset + "-" + name)

    if dataset == "WIKI":
        dataset = WIKI(data_home)
    cache = ComplexNumericalQueryDatasetCachePath(dataset.cache_path)
    data = NumericalComplexQueryData(dataset, cache_path=cache)
    data.preprocess_data_if_needed()
    data.load_cache([
        "meta",
    ])

    entity_count = data.entity_count
    relation_count = data.relation_count
    attribute_count = data.attribute_count
    value_count = data.value_count
    timestamp_count = data.timestamp_count
    max_relation_id = relation_count
    max_attribute_id = attribute_count
    model = NTFLEX(
        nentity=entity_count,
        nrelation=relation_count + max_relation_id,  # with reverse relations
        nattribute=attribute_count + max_attribute_id,
        nvalue=value_count,
        ntimestamp=timestamp_count,
        hidden_dim=hidden_dim,
        gamma=gamma,
        center_reg=center_reg,
        test_batch_size=test_batch_size,
        drop=input_dropout,
    )
    MyExperiment(
        output, data, model,
        start_step, max_steps, every_test_step, every_valid_step,
        batch_size, test_batch_size, negative_sample_size,
        train_device, test_device,
        resume, resume_by_score,
        lr, cpu_num,
        hidden_dim, input_dropout, gamma, center_reg, train_tasks, train_all, eval_tasks, eval_all
    )


if __name__ == '__main__':
    main()
