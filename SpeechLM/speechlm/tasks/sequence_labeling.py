# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import sys

from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq.data import Dictionary, data_utils, LanguagePairDataset
from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig
from fairseq.tasks.audio_finetuning import AudioFinetuningTask, AudioFinetuningConfig, LabelEncoder, label_len_fn
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from fairseq.tasks import register_task
from speechlm.data.add_sequential_label_dataset import AddSequentialLabelDataset

@dataclass
class AudioLabelingConfig(AudioFinetuningConfig):
    pass

@dataclass
class TextLabelingConfig(AudioLabelingConfig):
    text_data: str = field(default=MISSING, metadata={"help": "path to binary text data directory"})

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
