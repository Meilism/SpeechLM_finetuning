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
from collections import OrderedDict
import numpy as np

from fairseq.data import encoders, data_utils, Dictionary, LanguagePairDataset, FileAudioDataset
from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data.text_compressor import TextCompressionLevel

from fairseq.tasks import FairseqTask, register_task
from speechlm.data.add_label_dataset import AddLabelDataset
from speechlm.data.concat_dataset import ConcatDataset
from speechlm.data.sampled_speech_text_dataset import SampledSpeechTextDataset


logger = logging.getLogger(__name__)


@dataclass
class AudioClassificationConfig(AudioPretrainingConfig):
    pass

@dataclass
class TextClassificationConfig(AudioPretrainingConfig):
    text_data: str = field(default=MISSING, metadata={"help": "path to binary text data directory"})

@dataclass
class MixedClassificationConfig(TextClassificationConfig):
    sampling_temp: int = field(default=1, metadata={"help": "sampling temperature, 1 means true data distribution and 100 means (almost) uniform distribution"})

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
        self, split: str, task_cfg: TextClassificationConfig = None, **kwargs
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

@register_task("slue_mixed_classification", dataclass=MixedClassificationConfig)
class MixedClassificationTask(TextClassificationTask):
    """ """

    cfg: MixedClassificationConfig

    def __init__(
        self,
        cfg: MixedClassificationConfig,
    ):
        super().__init__(cfg)
        self.audio_datasets = {}
        self.text_datasets = {}
        self.blank_symbol = "<s>"

        self.state.add_factory("label2id", self.load_label2id)

    def load_audio_dataset(self, split: str, task_cfg: MixedClassificationConfig = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

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
        
        manifest_path = os.path.join(data_path, "{}.tsv".format(split))
        self.audio_datasets[split] = FileAudioDataset(
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                text_compression_level=text_compression_level,
                compute_mask=compute_mask,
                **mask_args,
            )

        if getattr(task_cfg, "subsample", 1) < 1:
            self.audio_datasets[split] = SubsampleDataset(
                self.audio_datasets[split],
                task_cfg.subsample,
                shuffle=True,
                seed=task_cfg.seed,
            )

        assert task_cfg.labels is not None
        data_path = self.cfg.data
        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.audio_datasets[split], "skipped_indices", set())
        logger.info(f"label2id: {self.label2id}")
        with open(label_path, "r") as f:
            labels = [
                self.label2id[l.strip()]
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]

        assert len(labels) == len(self.audio_datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.audio_datasets[split])}) do not match"
        )

        self.audio_datasets[split] = AddLabelDataset(
            self.audio_datasets[split],
            labels,
        )

    def load_text_dataset(
        self, split: str, task_cfg: MixedClassificationConfig = None, **kwargs
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
        text_dataset = data_utils.load_indexed_dataset(text_data_path, text_dictionary, "lazy")
        text_dataset = LanguagePairDataset(text_dataset, text_dataset.sizes, text_dictionary, left_pad_source=False, shuffle=True)
        self.text_datasets[split] = text_dataset

        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.text_datasets[split], "skipped_indices", set())
        logger.info(f"label2id: {self.label2id}")
        with open(label_path, "r") as f:
            labels = [
                self.label2id[l.strip()]
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]

        assert len(labels) == len(self.text_datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.text_datasets[split])}) do not match"
        )

        self.text_datasets[split] = AddLabelDataset(
            self.text_datasets[split],
            labels,
        )
    
    def load_dataset(
        self, split: str, task_cfg: MixedClassificationConfig = None, **kwargs
    ):
        text_split = '-'.join(split.split('-')[:2])
        self.load_audio_dataset(split, task_cfg, **kwargs)
        self.load_text_dataset(text_split, task_cfg, **kwargs)
        ratios = np.array([len(self.audio_datasets[split]), len(self.text_datasets[text_split])]) / \
            (len(self.audio_datasets[split]) + len(self.text_datasets[text_split]))
        ratios = ratios ** (1/self.cfg.sampling_temp)
        ratios = ratios / sum(ratios)

        def randnum_generator(x):
            return np.random.choice([0, 1], size=1, p=ratios).item()
      
        self.datasets[split] = SampledSpeechTextDataset(
            OrderedDict(
                [(0, self.audio_datasets[split]), (1, self.text_datasets[text_split])]
            ),
            randnum_generator
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize