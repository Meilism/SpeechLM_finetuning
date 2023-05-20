# Copyright (c) ASAPP Inc.
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq.data import encoders, data_utils, Dictionary, LanguagePairDataset
from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data.text_compressor import TextCompressionLevel

from fairseq.tasks import FairseqTask, register_task
from speechlm.data.add_label_dataset import AddLabelDataset
from speechlm.data.concat_dataset import ConcatDataset


logger = logging.getLogger(__name__)


@dataclass
class AudioClassificationConfig(AudioPretrainingConfig):
    pass

@dataclass
class TextClassificationConfig(AudioPretrainingConfig):
    text_data: str = field(default=MISSING, metadata={"help": "path to binary text data directory"})

# add slue_ as prefix of the registerred name in case there are conflicts in future
@register_task("slue_audio_classification", dataclass=AudioClassificationConfig)
class AudioClassificationTask(AudioPretrainingTask):
    """ """

    cfg: AudioClassificationConfig

    def __init__(
        self,
        cfg: AudioClassificationConfig,
    ):
        super().__init__(cfg)
        self.blank_symbol = "<s>"

        self.state.add_factory("label2id", self.load_label2id)

    def load_label2id(self):
        assert self.cfg.labels
        dict_path = os.path.join(self.cfg.data, f"labels.{self.cfg.labels}.txt")
        with open(dict_path) as f:
            labels = [line.strip() for line in f]
        label2id = {l: i for i, l in enumerate(labels)}
        return label2id

    def load_dataset(
        self, split: str, task_cfg: AudioClassificationConfig = None, **kwargs
    ):
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None
        data_path = self.cfg.data
        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        logger.info(f"label2id: {self.label2id}")
        with open(label_path, "r") as f:
            labels = [
                self.label2id[l.strip()]
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        self.datasets[split] = AddLabelDataset(
            self.datasets[split],
            labels,
        )

    @property
    def label2id(self):
        return self.state.label2id

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)


@register_task("slue_text_classification", dataclass=TextClassificationConfig)
class TextClassificationTask(FairseqTask):
    """ """

    cfg: TextClassificationConfig

    def __init__(
        self,
        cfg: TextClassificationConfig,
    ):
        super().__init__(cfg)
        self.blank_symbol = "<s>"

        self.state.add_factory("label2id", self.load_label2id)

    @classmethod
    def setup_task(cls, cfg: AudioPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_label2id(self):
        assert self.cfg.labels
        dict_path = os.path.join(self.cfg.data, f"labels.{self.cfg.labels}.txt")
        with open(dict_path) as f:
            labels = [line.strip() for line in f]
        label2id = {l: i for i, l in enumerate(labels)}
        return label2id

    def load_dataset(
        self, split: str, task_cfg: AudioClassificationConfig = None, **kwargs
    ):
        data_path = self.cfg.data
        text_data_dir = self.cfg.text_data
        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )

        compute_mask = task_cfg.precompute_mask_config is not None
        mask_args = {}
        if compute_mask:
            mask_args = task_cfg.precompute_mask_config
            
        text_dictionary = Dictionary.load(f"{text_data_dir}/dict.phn.txt")
        text_data_path = os.path.join(text_data_dir, "{}_upsample.phn-ltr.phn".format(split))
        text_dataset = data_utils.load_indexed_dataset(text_data_path, text_dictionary,"lazy")
        text_dataset = LanguagePairDataset(text_dataset, text_dataset.sizes, text_dictionary, left_pad_source=False, shuffle=True)
        self.datasets[split] = text_dataset

        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        logger.info(f"label2id: {self.label2id}")
        with open(label_path, "r") as f:
            labels = [
                self.label2id[l.strip()]
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        self.datasets[split] = AddLabelDataset(
            self.datasets[split],
            labels,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

    @property
    def label2id(self):
        return self.state.label2id

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(model_cfg, from_checkpoint)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            # if "w2v_args" in actualized_cfg:
            if hasattr(actualized_cfg, "w2v_args"):
                model_cfg.w2v_args = actualized_cfg.w2v_args

        return model

    def post_save(self, cp_path, num_updates):
        if self.cfg.post_save_script is not None:
            logger.info(f"launching {self.cfg.post_save_script}")
            import os.path as osp
            from fairseq.file_io import PathManager

            eval_cp_path = osp.join(
                osp.dirname(cp_path), f"checkpoint_eval_{num_updates}.pt"
            )

            print(cp_path, eval_cp_path, osp.dirname(cp_path))

            assert PathManager.copy(
                cp_path, eval_cp_path, overwrite=True
            ), f"Failed to copy {cp_path} to {eval_cp_path}"

            import subprocess
            import shlex

            subprocess.call(shlex.split(f"{self.cfg.post_save_script} {eval_cp_path}"))
