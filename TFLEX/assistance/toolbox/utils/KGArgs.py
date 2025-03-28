import json
import os
from enum import Enum
from typing import Optional, Dict, Any, List

import torch
from dataclasses import dataclass, field, asdict

from ..utils.Framework import is_torch_available, cached_property, torch_required


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class ParallelMode(Enum):
    NOT_PARALLEL = "not_parallel"
    NOT_DISTRIBUTED = "not_distributed"
    DISTRIBUTED = "distributed"


@dataclass
class DataLoaderArguments:
    train_batch_size: int = field(default=512, metadata={"help": "batch size"})
    train_shuffle: bool = field(default=False, metadata={"help": "shuffle data"})
    train_drop_last: bool = field(default=True, metadata={"help": "drop last batch"})
    train_num_workers: int = field(default=2, metadata={"help": "number of workers"})
    train_pin_memory: bool = field(default=False, metadata={"help": "pin memory"})

    valid_batch_size: int = field(default=512, metadata={"help": "batch size"})
    valid_shuffle: bool = field(default=False, metadata={"help": "shuffle data"})
    valid_drop_last: bool = field(default=True, metadata={"help": "drop last batch"})
    valid_num_workers: int = field(default=2, metadata={"help": "number of workers"})
    valid_pin_memory: bool = field(default=False, metadata={"help": "pin memory"})

    test_batch_size: int = field(default=512, metadata={"help": "batch size"})
    test_shuffle: bool = field(default=False, metadata={"help": "shuffle data"})
    test_drop_last: bool = field(default=True, metadata={"help": "drop last batch"})
    test_num_workers: int = field(default=2, metadata={"help": "number of workers"})
    test_pin_memory: bool = field(default=False, metadata={"help": "pin memory"})


@dataclass
class OutputArguments:
    """
    Parameters:
            output_dir (:obj:`str`):
                The output directory where the model predictions and checkpoints will be written.
            overwrite_output_dir (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`True`, overwrite the content of the output directory. Use this to continue training if
                :obj:`output_dir` points to a checkpoint directory.
    """
    output_dir: str = field(default="output", metadata={
        "help": "The output directory where the model predictions and checkpoints will be written."
    })
    overwrite_output_dir: bool = field(default=False, metadata={
        "help": (
            "Overwrite the content of the output directory."
            "Use this to continue training if output_dir points to a checkpoint directory."
        )
    })


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.

    Parameters:
        do_train (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run training or not. This argument is not directly used by :class:`~transformers.Trainer`, it's
            intended to be used by your training/evaluation scripts instead. See the `example scripts
            <https://github.com/huggingface/transformers/tree/master/examples>`__ for more details.
        do_eval (:obj:`bool`, `optional`):
            Whether to run evaluation on the validation set or not. Will be set to :obj:`True` if
            :obj:`evaluation_strategy` is different from :obj:`"no"`. This argument is not directly used by
            :class:`~transformers.Trainer`, it's intended to be used by your training/evaluation scripts instead. See
            the `example scripts <https://github.com/huggingface/transformers/tree/master/examples>`__ for more
            details.
        do_predict (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run predictions on the test set or not. This argument is not directly used by
            :class:`~transformers.Trainer`, it's intended to be used by your training/evaluation scripts instead. See
            the `example scripts <https://github.com/huggingface/transformers/tree/master/examples>`__ for more
            details.
        evaluation_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"no"`):
            The evaluation strategy to adopt during training. Possible values are:

                * :obj:`"no"`: No evaluation is done during training.
                * :obj:`"steps"`: Evaluation is done (and logged) every :obj:`eval_steps`.
                * :obj:`"epoch"`: Evaluation is done at the end of each epoch.

        prediction_loss_only (:obj:`bool`, `optional`, defaults to `False`):
            When performing evaluation and generating predictions, only returns the loss.
        per_device_train_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for training.
        per_device_eval_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for evaluation.
        gradient_accumulation_steps (:obj:`int`, `optional`, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

            .. warning::

                When using gradient accumulation, one step is counted as one step with backward pass. Therefore,
                logging, evaluation, save will be conducted every ``gradient_accumulation_steps * xxx_step`` training
                examples.
        eval_accumulation_steps (:obj:`int`, `optional`):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
            requires more memory).
        learning_rate (:obj:`float`, `optional`, defaults to 5e-5):
            The initial learning rate for :class:`~transformers.AdamW` optimizer.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in
            :class:`~transformers.AdamW` optimizer.
        adam_beta1 (:obj:`float`, `optional`, defaults to 0.9):
            The beta1 hyperparameter for the :class:`~transformers.AdamW` optimizer.
        adam_beta2 (:obj:`float`, `optional`, defaults to 0.999):
            The beta2 hyperparameter for the :class:`~transformers.AdamW` optimizer.
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            The epsilon hyperparameter for the :class:`~transformers.AdamW` optimizer.
        max_grad_norm (:obj:`float`, `optional`, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(:obj:`float`, `optional`, defaults to 3.0):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
            the last epoch before stopping training).
        max_steps (:obj:`int`, `optional`, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides
            :obj:`num_train_epochs`.
        lr_scheduler_type (:obj:`str` or :class:`~transformers.SchedulerType`, `optional`, defaults to :obj:`"linear"`):
            The scheduler type to use. See the documentation of :class:`~transformers.SchedulerType` for all possible
            values.
        warmup_ratio (:obj:`float`, `optional`, defaults to 0.0):
            Ratio of total training steps used for a linear warmup from 0 to :obj:`learning_rate`.
        warmup_steps (:obj:`int`, `optional`, defaults to 0):
            Number of steps used for a linear warmup from 0 to :obj:`learning_rate`. Overrides any effect of
            :obj:`warmup_ratio`.
        logging_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"steps"`):
            The logging strategy to adopt during training. Possible values are:

                * :obj:`"no"`: No logging is done during training.
                * :obj:`"epoch"`: Logging is done at the end of each epoch.
                * :obj:`"steps"`: Logging is done every :obj:`logging_steps`.

        logging_first_step (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to log and evaluate the first :obj:`global_step` or not.
        logging_steps (:obj:`int`, `optional`, defaults to 500):
            Number of update steps between two logs if :obj:`logging_strategy="steps"`.
        save_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"steps"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                * :obj:`"no"`: No save is done during training.
                * :obj:`"epoch"`: Save is done at the end of each epoch.
                * :obj:`"steps"`: Save is done every :obj:`save_steps`.

        save_steps (:obj:`int`, `optional`, defaults to 500):
            Number of updates steps before two checkpoint saves if :obj:`save_strategy="steps"`.
        save_total_limit (:obj:`int`, `optional`):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            :obj:`output_dir`.
        no_cuda (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to not use CUDA even when it is available or not.
        seed (:obj:`int`, `optional`, defaults to 42):
            Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
            :func:`~transformers.Trainer.model_init` function to instantiate the model if it has some randomly
            initialized parameters.
        fp16 (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use 16-bit (mixed) precision training instead of 32-bit training.
        fp16_opt_level (:obj:`str`, `optional`, defaults to 'O1'):
            For :obj:`fp16` training, Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details
            on the `Apex documentation <https://nvidia.github.io/apex/amp.html>`__.
        fp16_backend (:obj:`str`, `optional`, defaults to :obj:`"auto"`):
            The backend to use for mixed precision training. Must be one of :obj:`"auto"`, :obj:`"amp"` or
            :obj:`"apex"`. :obj:`"auto"` will use AMP or APEX depending on the PyTorch version detected, while the
            other choices will force the requested backend.
        fp16_full_eval (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use full 16-bit precision evaluation instead of 32-bit. This will be faster and save memory but
            can harm metric values.
        local_rank (:obj:`int`, `optional`, defaults to -1):
            Rank of the process during distributed training.
        tpu_num_cores (:obj:`int`, `optional`):
            When training on TPU, the number of TPU cores (automatically passed by launcher script).
        debug (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When training on TPU, whether to print debug metrics or not.
        dataloader_drop_last (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (:obj:`int`, `optional`):
            Number of update steps between two evaluations if :obj:`evaluation_strategy="steps"`. Will default to the
            same value as :obj:`logging_steps` if not set.
        dataloader_num_workers (:obj:`int`, `optional`, defaults to 0):
            Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
            main process.
        past_index (:obj:`int`, `optional`, defaults to -1):
            Some models like :doc:`TransformerXL <../model_doc/transformerxl>` or :doc`XLNet <../model_doc/xlnet>` can
            make use of the past hidden states for their predictions. If this argument is set to a positive int, the
            ``Trainer`` will use the corresponding output (usually index 2) as the past state and feed it to the model
            at the next training step under the keyword argument ``mems``.
        run_name (:obj:`str`, `optional`):
            A descriptor for the run. Typically used for `wandb <https://www.wandb.com/>`_ logging.
        remove_unused_columns (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If using :obj:`datasets.Dataset` datasets, whether or not to automatically remove the columns unused by the
            model forward method.

            (Note that this behavior is not implemented for :class:`~transformers.TFTrainer` yet.)
        label_names (:obj:`List[str]`, `optional`):
            The list of keys in your dictionary of inputs that correspond to the labels.

            Will eventually default to :obj:`["labels"]` except if the model used is one of the
            :obj:`XxxForQuestionAnswering` in which case it will default to :obj:`["start_positions",
            "end_positions"]`.
        load_best_model_at_end (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to load the best model found during training at the end of training.

            .. note::

                When set to :obj:`True`, the parameters :obj:`save_strategy` and :obj:`save_steps` will be ignored and
                the model will be saved after each evaluation.
        metric_for_best_model (:obj:`str`, `optional`):
            Use in conjunction with :obj:`load_best_model_at_end` to specify the metric to use to compare two different
            models. Must be the name of a metric returned by the evaluation with or without the prefix :obj:`"eval_"`.
            Will default to :obj:`"loss"` if unspecified and :obj:`load_best_model_at_end=True` (to use the evaluation
            loss).

            If you set this value, :obj:`greater_is_better` will default to :obj:`True`. Don't forget to set it to
            :obj:`False` if your metric is better when lower.
        greater_is_better (:obj:`bool`, `optional`):
            Use in conjunction with :obj:`load_best_model_at_end` and :obj:`metric_for_best_model` to specify if better
            models should have a greater metric or not. Will default to:

            - :obj:`True` if :obj:`metric_for_best_model` is set to a value that isn't :obj:`"loss"` or
              :obj:`"eval_loss"`.
            - :obj:`False` if :obj:`metric_for_best_model` is not set, or set to :obj:`"loss"` or :obj:`"eval_loss"`.
        ignore_data_skip (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When resuming training, whether or not to skip the epochs and batches to get the data loading at the same
            stage as in the previous training. If set to :obj:`True`, the training will begin faster (as that skipping
            step can take a long time) but will not yield the same results as the interrupted training would have.
        sharded_ddp (:obj:`bool`, :obj:`str` or list of :class:`~transformers.trainer_utils.ShardedDDPOption`, `optional`, defaults to :obj:`False`):
            Use Sharded DDP training from `FairScale <https://github.com/facebookresearch/fairscale>`__ (in distributed
            training only). This is an experimental feature.

            A list of options along the following:

            - :obj:`"simple"`: to use first instance of sharded DDP released by fairscale (:obj:`ShardedDDP`) similar
              to ZeRO-2.
            - :obj:`"zero_dp_2"`: to use the second instance of sharded DPP released by fairscale
              (:obj:`FullyShardedDDP`) in Zero-2 mode (with :obj:`reshard_after_forward=False`).
            - :obj:`"zero_dp_3"`: to use the second instance of sharded DPP released by fairscale
              (:obj:`FullyShardedDDP`) in Zero-3 mode (with :obj:`reshard_after_forward=True`).
            - :obj:`"offload"`: to add ZeRO-offload (only compatible with :obj:`"zero_dp_2"` and :obj:`"zero_dp_3"`).

            If a string is passed, it will be split on space. If a bool is passed, it will be converted to an empty
            list for :obj:`False` and :obj:`["simple"]` for :obj:`True`.
        deepspeed (:obj:`str` or :obj:`dict`, `optional`):
            Use `Deepspeed <https://github.com/microsoft/deepspeed>`__. This is an experimental feature and its API may
            evolve in the future. The value is either the location of DeepSpeed json config file (e.g.,
            ``ds_config.json``) or an already loaded json file as a :obj:`dict`"
        label_smoothing_factor (:obj:`float`, `optional`, defaults to 0.0):
            The label smoothing factor to use. Zero means no label smoothing, otherwise the underlying onehot-encoded
            labels are changed from 0s and 1s to :obj:`label_smoothing_factor/num_labels` and :obj:`1 -
            label_smoothing_factor + label_smoothing_factor/num_labels` respectively.
        adafactor (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the :class:`~transformers.Adafactor` optimizer instead of
            :class:`~transformers.AdamW`.
        group_by_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to group together samples of roughly the same length in the training dataset (to minimize
            padding applied and be more efficient). Only useful if applying dynamic padding.
        length_column_name (:obj:`str`, `optional`, defaults to :obj:`"length"`):
            Column name for precomputed lengths. If the column exists, grouping by length will use these values rather
            than computing them on train startup. Ignored unless :obj:`group_by_length` is :obj:`True` and the dataset
            is an instance of :obj:`Dataset`.
        ddp_find_unused_parameters (:obj:`bool`, `optional`):
            When using distributed training, the value of the flag :obj:`find_unused_parameters` passed to
            :obj:`DistributedDataParallel`. Will default to :obj:`False` if gradient checkpointing is used, :obj:`True`
            otherwise.
        dataloader_pin_memory (:obj:`bool`, `optional`, defaults to :obj:`True`)):
            Whether you want to pin memory in data loaders or not. Will default to :obj:`True`.
        skip_memory_metrics (:obj:`bool`, `optional`, defaults to :obj:`False`)):
            Whether to skip adding of memory profiler reports to metrics. Defaults to :obj:`False`.

    """

    do_train: bool = field(default=False, metadata={
        "help": "Whether to run training."
    })
    do_eval: bool = field(default=None, metadata={
        "help": "Whether to run eval on the dev set."
    })
    do_predict: bool = field(default=False, metadata={
        "help": "Whether to run predictions on the test set."
    })
    prediction_loss_only: bool = field(default=False, metadata={
        "help": "When performing evaluation and predictions, only returns the loss."
    })
    resume_training: bool = field(default=False, metadata={
        "help": "if it do train, it will try to load checkpoint firstly."
    })

    per_device_train_batch_size: int = field(default=8, metadata={
        "help": "Batch size per GPU/TPU core/CPU for training."
    })
    per_device_eval_batch_size: int = field(default=8, metadata={
        "help": "Batch size per GPU/TPU core/CPU for evaluation."
    })

    gradient_accumulation_steps: int = field(default=1, metadata={
        "help": "Number of updates steps to accumulate before performing a backward/update pass."
    })
    eval_accumulation_steps: Optional[int] = field(default=None, metadata={
        "help": "Number of predictions steps to accumulate before moving the tensors to the CPU."
    })

    learning_rate: float = field(default=5e-5, metadata={
        "help": "The initial learning rate for AdamW."
    })
    weight_decay: float = field(default=0.0, metadata={
        "help": "Weight decay for AdamW if we apply some."
    })
    adam_beta1: float = field(default=0.9, metadata={
        "help": "Beta1 for AdamW optimizer"
    })
    adam_beta2: float = field(default=0.999, metadata={
        "help": "Beta2 for AdamW optimizer"
    })
    adam_epsilon: float = field(default=1e-8, metadata={
        "help": "Epsilon for AdamW optimizer."
    })
    max_grad_norm: float = field(default=1.0, metadata={
        "help": "Max gradient norm."
    })

    num_train_epochs: float = field(default=3.0, metadata={
        "help": "Total number of training epochs to perform."
    })
    max_steps: int = field(default=-1, metadata={
        "help": "If > 0: set total number of training steps to perform. Override num_train_epochs."
    })
    lr_scheduler_type: SchedulerType = field(default="linear", metadata={
        "help": "The scheduler type to use."
    })
    warmup_ratio: float = field(default=0.0, metadata={
        "help": "Linear warmup over warmup_ratio fraction of total steps."
    })
    warmup_steps: int = field(default=0, metadata={
        "help": "Linear warmup over warmup_steps."
    })

    logging_strategy: IntervalStrategy = field(default="steps", metadata={
        "help": "The logging strategy to use."
    })
    logging_first_step: bool = field(default=False, metadata={
        "help": "Log the first global_step"
    })
    logging_steps: int = field(default=500, metadata={
        "help": "Log every X updates steps."
    })
    save_strategy: IntervalStrategy = field(default="steps", metadata={
        "help": "The checkpoint save strategy to use."
    })
    save_steps: int = field(default=500, metadata={
        "help": "Save checkpoint every X updates steps."
    })
    save_total_limit: Optional[int] = field(default=None, metadata={
        "help": (
            "Limit the total amount of checkpoints."
            "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
        )
    })
    seed: int = field(default=123, metadata={
        "help": "Random seed that will be set at the beginning of training."
    })
    no_cuda: bool = field(default=False, metadata={
        "help": "Do not use CUDA even when it is available"
    })

    fp16: bool = field(default=False, metadata={
        "help": "Whether to use 16-bit (mixed) precision instead of 32-bit"
    })
    fp16_opt_level: str = field(default="O1", metadata={
        "help": (
            "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html"
        )
    })
    fp16_backend: str = field(default="auto", metadata={
        "help": "The backend to be used for mixed precision.", "choices": ["auto", "amp", "apex"]
    })
    fp16_full_eval: bool = field(default=False, metadata={
        "help": "Whether to use full 16-bit precision evaluation instead of 32-bit"
    })
    local_rank: int = field(default=-1, metadata={
        "help": "For distributed training: local_rank"
    })

    tpu_num_cores: Optional[int] = field(default=None, metadata={
        "help": "TPU: Number of TPU cores (automatically passed by launcher script)"
    })
    tpu_metrics_debug: bool = field(default=False, metadata={
        "help": "Deprecated, the use of `--debug` is preferred. TPU: Whether to print debug metrics"
    })
    debug: bool = field(default=False, metadata={
        "help": "Whether to print debug metrics on TPU"
    })

    dataloader_drop_last: bool = field(default=False, metadata={
        "help": "Drop the last incomplete batch if it is not divisible by the batch size."
    })
    eval_steps: int = field(default=None, metadata={
        "help": "Run an evaluation every X steps."
    })
    dataloader_num_workers: int = field(default=0, metadata={
        "help": "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process."
    })

    past_index: int = field(default=-1, metadata={
        "help": "If >=0, uses the corresponding part of the output as the past state for next step."
    })

    remove_unused_columns: Optional[bool] = field(default=True, metadata={
        "help": "Remove columns not required by the model when using an nlp.Dataset."
    })
    label_names: Optional[List[str]] = field(default=None, metadata={
        "help": "The list of keys in your dictionary of inputs that correspond to the labels."
    })

    load_best_model_at_end: Optional[bool] = field(default=False, metadata={
        "help": "Whether or not to load the best model found during training at the end of training."
    })
    metric_for_best_model: Optional[str] = field(default=None, metadata={
        "help": "The metric to use to compare two different models."
    })
    greater_is_better: Optional[bool] = field(default=None, metadata={
        "help": "Whether the `metric_for_best_model` should be maximized or not."
    })
    ignore_data_skip: bool = field(default=False, metadata={
        "help": "When resuming training, whether or not to skip the first epochs and batches to get to the same training data."
    })
    sharded_ddp: str = field(default="", metadata={
        "help": "Whether or not to use sharded DDP training (in distributed training only). The base option "
                "should be `simple`, `zero_dp_2` or `zero_dp_3` and you can add CPU-offload to `zero_dp_2` or `zero_dp_3` "
                "like this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or "
                "with the same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`.",
    })
    deepspeed: Optional[str] = field(default=None, metadata={
        "help": "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already loaded json file as a dict"
    })
    label_smoothing_factor: float = field(default=0.0, metadata={
        "help": "The label smoothing epsilon to apply (zero means no label smoothing)."
    })
    adafactor: bool = field(default=False, metadata={
        "help": "Whether or not to replace AdamW by Adafactor."
    })
    group_by_length: bool = field(default=False, metadata={
        "help": "Whether or not to group samples of roughly the same length together when batching."
    })
    length_column_name: Optional[str] = field(default="length", metadata={
        "help": "Column name with precomputed lengths to use when grouping by length."
    })
    ddp_find_unused_parameters: Optional[bool] = field(default=None, metadata={
        "help": "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
    })
    dataloader_pin_memory: bool = field(default=True, metadata={
        "help": "Whether or not to pin memory for DataLoader."
    })
    skip_memory_metrics: bool = field(default=False, metadata={
        "help": "Whether or not to skip adding of memory profiler reports to metrics."
    })
    _n_gpu: int = field(init=False, repr=False, default=-1)
    mp_parameters: str = field(default="", metadata={
        "help": "Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer"
    })

    def __post_init__(self):
        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        #  see https://github.com/huggingface/transformers/issues/10628

        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.eval_steps is None:
            self.eval_steps = self.logging_steps

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]

        if is_torch_available() and self.device.type != "cuda" and (self.fp16 or self.fp16_full_eval):
            raise ValueError(
                "Mixed precision training with AMP or APEX (`--fp16`) and FP16 evaluation can only be used on CUDA devices."
            )

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            print(
                "Both warmup_ratio and warmup_steps given",
                "warmup_steps will override any effect of warmup_ratio during training"
            )

    def __repr__(self):
        # We override the default repr to remove deprecated arguments from the repr. This method should be removed once
        # those deprecated arguments are removed form TrainingArguments. (TODO: v5)
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v}" for k, v in self_as_dict.items()]
        return f"{self.__class__.__name__}({', '.join(attrs_as_str)})"

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from :obj:`per_device_train_batch_size` in distributed training).
        """
        per_device_batch_size = self.per_device_train_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_device_eval_batch_size` in distributed training).
        """
        per_device_batch_size = self.per_device_eval_batch_size
        eval_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return eval_batch_size

    @property
    def place_model_on_device(self):
        """
        Can be subclassed and overridden for some specific integrations.
        """
        return True

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        print("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif self.deepspeed:
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            from .Framework import is_deepspeed_available

            if not is_deepspeed_available():
                raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
            import deepspeed

            deepspeed.init_distributed()

            # workaround for setups like notebooks where the launcher can't be used,
            # but deepspeed requires a dist env.
            # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
            self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    @torch_required
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu

    @property
    @torch_required
    def parallel_mode(self):
        """
        The current mode used for parallelism if multiple GPUs/TPU cores are available. One of:

        - :obj:`ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - :obj:`ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses :obj:`torch.nn.DataParallel`).
        - :obj:`ParallelMode.DISTRIBUTED`: several GPUs, each having its own process (uses
          :obj:`torch.nn.DistributedDataParallel`).
        """
        if self.local_rank != -1:
            return ParallelMode.DISTRIBUTED
        elif self.n_gpu > 1:
            return ParallelMode.NOT_DISTRIBUTED
        else:
            return ParallelMode.NOT_PARALLEL

    @property
    @torch_required
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        if self.local_rank != -1:
            return torch.distributed.get_world_size()
        return 1

    @property
    @torch_required
    def process_index(self):
        """
        The number of processes used in parallel.
        """
        if self.local_rank != -1:
            return torch.distributed.get_rank()
        return 0

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}
