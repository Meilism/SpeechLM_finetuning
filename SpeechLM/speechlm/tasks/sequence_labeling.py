# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import sys

import numpy as np
from dataclasses import dataclass, field
from omegaconf import MISSING
from collections import OrderedDict

from fairseq.data import Dictionary, data_utils, LanguagePairDataset, AddTargetDataset
from fairseq.tasks.audio_finetuning import AudioFinetuningTask, AudioFinetuningConfig, LabelEncoder, label_len_fn
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from fairseq.tasks import register_task
from speechlm.data.add_sequential_label_dataset import AddSequentialLabelDataset
from speechlm.data.sampled_speech_text_dataset import SampledSpeechTextDataset

@dataclass
class AudioLabelingConfig(AudioFinetuningConfig):
    pass

@dataclass
class TextLabelingConfig(AudioLabelingConfig):
    text_data: str = field(default=MISSING, metadata={"help": "path to binary text data directory"})

@dataclass
class MixedLabelingConfig(TextLabelingConfig):
    sampling_temp: int = field(default=1, metadata={"help": "sampling temperature, 1 means true data distribution and 100 means (almost) uniform distribution"})

@register_task("slue_audio_labeling", dataclass=AudioLabelingConfig)
class AudioLabelingTask(AudioFinetuningTask):
    """ """

    cfg: AudioFinetuningConfig

    def __init__(
        self,
        cfg: AudioFinetuningConfig,
    ):
        super().__init__(cfg)

@register_task("slue_text_labeling", dataclass=TextLabelingConfig)
class TextLabelingTask(AudioFinetuningTask):
    """ """

    cfg: TextLabelingConfig

    def __init__(
        self,
        cfg: TextLabelingConfig,
    ):
        super().__init__(cfg)

    def load_dataset(
        self, split: str, task_cfg: TextLabelingConfig = None, **kwargs
    ):
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None
        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        data_path = self.cfg.data
        text_data_dir = self.cfg.text_data

        text_dictionary = Dictionary.load(f"{text_data_dir}/dict.phn.txt")
        text_data_path = os.path.join(text_data_dir, "{}_upsample.phn-ltr.phn".format(split))
        text_dataset = data_utils.load_indexed_dataset(text_data_path, text_dictionary,"lazy")
        text_dataset = LanguagePairDataset(text_dataset, text_dataset.sizes, text_dictionary, left_pad_source=False, shuffle=False)
        self.datasets[split] = text_dataset

        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        text_compressor = TextCompressor(level=text_compression_level)
        with open(label_path, "r") as f:
            labels = [
                text_compressor.compress(l)
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        process_label = LabelEncoder(self.target_dictionary)

        self.datasets[split] = AddSequentialLabelDataset(
            self.datasets[split],
            labels,
            pad=self.target_dictionary.pad(),
            eos=self.target_dictionary.eos(),
            batch_targets=True,
            process_label=process_label,
            label_len_fn=label_len_fn,
            add_to_input=task_cfg.get("autoregressive", False),
            text_compression_level=text_compression_level,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

@register_task("slue_mixed_labeling", dataclass=MixedLabelingConfig)
class MixedLabelingTask(AudioFinetuningTask):
    """ """

    cfg: MixedLabelingConfig

    def __init__(
        self,
        cfg: MixedLabelingConfig,
    ):
        super().__init__(cfg)
        self.audio_datasets = {}
        self.text_datasets = {}

    def load_audio_dataset(self, split: str, task_cfg: MixedLabelingConfig = None, **kwargs):
        super().load_dataset(split, task_cfg, **kwargs)

        self.audio_datasets[split] = self.datasets[split]

    def load_text_dataset(
        self, split: str, task_cfg: MixedLabelingConfig = None, **kwargs
    ):
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None
        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        data_path = self.cfg.data
        text_data_dir = self.cfg.text_data

        text_dictionary = Dictionary.load(f"{text_data_dir}/dict.phn.txt")
        text_data_path = os.path.join(text_data_dir, "{}_upsample.phn-ltr.phn".format(split))
        text_dataset = data_utils.load_indexed_dataset(text_data_path, text_dictionary,"lazy")
        text_dataset = LanguagePairDataset(text_dataset, text_dataset.sizes, text_dictionary, left_pad_source=False, shuffle=False)
        self.text_datasets[split] = text_dataset

        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.text_datasets[split], "skipped_indices", set())
        text_compressor = TextCompressor(level=text_compression_level)
        with open(label_path, "r") as f:
            labels = [
                text_compressor.compress(l)
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]

        assert len(labels) == len(self.text_datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.text_datasets[split])}) do not match"
        )

        process_label = LabelEncoder(self.target_dictionary)

        self.text_datasets[split] = AddSequentialLabelDataset(
            self.text_datasets[split],
            labels,
            pad=self.target_dictionary.pad(),
            eos=self.target_dictionary.eos(),
            batch_targets=True,
            process_label=process_label,
            label_len_fn=label_len_fn,
            add_to_input=task_cfg.get("autoregressive", False),
            text_compression_level=text_compression_level,
        )

    
    def load_dataset(
        self, split: str, task_cfg: MixedLabelingConfig = None, **kwargs
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