"""
@date: 2022/2/19
@description: 模型保存和恢复
"""
import os
from typing import Tuple, List, Union, Optional

import torch
from pathlib import Path
from torch import nn
from torch.optim import optimizer

from ..exp.OutputSchema import OutputPathSchema

_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_MODEL_STATE_DICT2 = "model_state_dict2"
_OPTIMIZER_STATE_DICT2 = "optimizer_state_dict2"
_EPOCH = "epoch"
_STEP = "step"
_BEST_SCORE = "best_score"
_LOSS = "loss"


def load_model(model: nn.Module, checkpoint_path="./output/temp/model.tar") -> float:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])
    best_score = checkpoint[_BEST_SCORE]
    return best_score


def save_model(model: nn.Module,
               best_score: float,
               save_path="./result/fr_en/model.tar"):
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _BEST_SCORE: best_score,
    }, save_path)


def load_checkpoint(model: nn.Module,
                    optim: Optional[optimizer.Optimizer],
                    checkpoint_path="./output/temp/checkpoint.tar") -> Tuple[int, int, float]:
    """Loads training checkpoint.

    :param checkpoint_path: path to checkpoint
    :param model: model to update state
    :param optim: optimizer to  update state
    :return: tuple of starting epoch id, starting step id, best checkpoint score
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])
    if optim is not None:
        optim.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])
    start_epoch_id = checkpoint[_EPOCH] + 1
    step = checkpoint[_STEP] + 1
    best_score = checkpoint[_BEST_SCORE]
    return start_epoch_id, step, best_score


def save_checkpoint(model: nn.Module,
                    optim: optimizer.Optimizer,
                    epoch_id: int,
                    step: int,
                    best_score: float,
                    save_path="./output/temp/checkpoint.tar"):
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optim.state_dict(),
        _EPOCH: epoch_id,
        _STEP: step,
        _BEST_SCORE: best_score,
    }, save_path)


def save_entity_embedding_list(entity_embedding, embedding_path="./output/temp/entity_embedding.txt"):
    with open(embedding_path, 'w') as f:
        d = entity_embedding.data.detach().cpu().numpy()
        for i in range(len(d)):
            f.write(" ".join([str(j) for j in d[i].tolist()]))
            f.write("\n")


class ModelParamStoreSchema:
    """模型参数保存，涉及
     1）检查点的保存和读取；
     2）用于部署的模型参数的保存和读取
    """

    def __init__(self, path: OutputPathSchema, best_checkpoint_filename="best_checkpoint.tar",
                 best_model_filename="best_model.tar"):
        self.path = path
        self.best_checkpoint_path = path.checkpoint_path(best_checkpoint_filename)
        self.best_model_path = path.deploy_path(best_model_filename)

    def save_best(self,
                  model: nn.Module,
                  optim: optimizer.Optimizer,
                  epoch_id: int,
                  step: int,
                  best_score: float,
                  ):
        # save model for training purpose
        save_checkpoint(model, optim, epoch_id, step, best_score, str(self.best_checkpoint_path))
        self.save_model_best(model, best_score)

    def load_best(self,
                  model: nn.Module,
                  optim: Optional[optimizer.Optimizer],
                  ) -> Tuple[int, int, float]:
        return load_checkpoint(model, optim, str(self.best_checkpoint_path))

    def checkpoint_path_with_score(self, score: float):
        return self.path.checkpoint_path("score-" + str(score) + "-checkpoint.tar")

    def model_path_with_score(self, score: float):
        return self.path.deploy_path("score-" + str(score) + "-checkpoint.tar")

    def model_pretrain_path(self):
        return self.path.deploy_path("pretrain.tar")

    def save_raw(self, anything, save_path):
        torch.save(anything, save_path)

    def load_raw(self, save_path):
        return torch.load(save_path)

    def save_scripts(self, filenames: List[Union[Path, str]]):
        filenames = list(set(filenames))
        for filename in filenames:
            with open(str(filename), "r", encoding="UTF-8") as source, open(str(self.path.scripts_path(Path(filename).name).resolve()), "w", encoding="UTF-8") as target:
                print("save", str(filename), "to", str(self.path.scripts_path(Path(filename).name).resolve()))
                target.writelines(source.readlines())

    def save_by_score(self,
                      model: nn.Module,
                      optim: optimizer.Optimizer,
                      epoch_id: int,
                      step: int,
                      score: float,
                      ):
        save_checkpoint(model, optim, epoch_id, step, score, str(self.checkpoint_path_with_score(score)))
        self.save_model_by_score(model, score)

    def load_by_score(self,
                      model: nn.Module,
                      optim: Optional[optimizer.Optimizer],
                      score: float,
                      ) -> Tuple[int, int, float]:
        return load_checkpoint(model, optim, str(self.checkpoint_path_with_score(score)))

    def save_model_best(self, model: nn.Module, best_score: float):
        # save model for deploy purpose
        save_model(model, best_score, str(self.best_model_path))

    def load_model_best(self, model: nn.Module) -> float:
        return load_model(model, str(self.best_model_path))

    def save_model_by_score(self, model: nn.Module, score: float):
        save_model(model, score, str(self.model_path_with_score(score)))

    def load_model_by_score(self, model: nn.Module, score: float) -> float:
        return load_model(model, str(self.model_path_with_score(score)))

    def delete_model_best(self):
        os.remove(str(self.best_model_path))

    def delete_model_by_score(self, score: float):
        os.remove(str(self.model_path_with_score(score)))
