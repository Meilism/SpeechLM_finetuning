# Copyright (c) ASAPP Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from omegaconf import MISSING, II
from typing import Any, Optional
from fairseq.models import BaseFairseqModel, register_model
from fairseq.tasks import FairseqTask

from fairseq.models.hubert import HubertAsrConfig, HubertEncoder
from fairseq.dataclass import FairseqDataclass
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES

@dataclass
class SpeechLMSeqClsConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to speechlm model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside speechlm model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside speechlm model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside speechlm model"
        },
    )
    conv_feature_layers: Optional[str] = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": (
                "string describing convolutional feature extraction "
                "layers in form of a python list that contains "
                "[(dim, kernel_size, stride), ...]"
            ),
        },
    )
    encoder_embed_dim: Optional[int] = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune speechlm for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in speechlm to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in speechlm"}
    )
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_channel_before: bool = False
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded wav2vec args
    w2v_args: Any = None

    # for sequence classification
    pool_method: str = field(default="avg", metadata={"help": "pooling method"})
    classifier_dropout: float = field(default=0.0, metadata={"help": "dropout"})


@register_model("speechlm_seq_cls", dataclass=SpeechLMSeqClsConfig)
class SpeechLMSeqCls(BaseFairseqModel):
    def __init__(self, cfg: SpeechLMSeqClsConfig, w2v_encoder: BaseFairseqModel, pooler: nn.Module, classifier: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.pooler = pooler
        self.classifier = classifier

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    @classmethod
    def build_model(cls, cfg: SpeechLMSeqClsConfig, task: FairseqTask):
        """Build a new model instance."""

        w2v_encoder = SpeechLMEncoder(cfg, task)
        w2v_encoder.w2v_model.mask_prob = cfg.mask_prob
        d = w2v_encoder.output_dim

        if cfg.pool_method == "avg":
            pooler = AvgPooler()
        elif cfg.pool_method == "self_attn":
            pooler = SelfAttnPooler(d)
        else:
            raise NotImplementedError(f"pooler_type={cfg.pool_method}")

        num_classes = len(task.label2id)
        classifier = nn.Sequential(
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(d, num_classes),
        )

        return cls(cfg, w2v_encoder, pooler, classifier)

    def get_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"].T][..., 0] = float("inf")
            logits[net_output["padding_mask"].T][..., 1:] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        padding_mask = (
            x["padding_mask"].transpose(0, 1) if x["padding_mask"] is not None else None
        )
        pooled = self.pooler(x["encoder_out"], padding_mask)
        pooled = self.classifier(pooled)
        x["pooled"] = pooled
        return x

    def extract_features(self, *args, **kwargs):
        return self.w2v_encoder.w2v_model.extract_features(*args, **kwargs)


class SpeechLMEncoder(HubertEncoder):
    def __init__(self, cfg: SpeechLMSeqClsConfig, task):
        super().__init__(cfg, task)
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)
        self.output_dim = w2v_args.model.encoder_embed_dim

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        results = super().forward(
            source,
            padding_mask,
            tbc,
            **kwargs,
        )
        return results


class AvgPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_out, padding_mask):
        """
        encoder_out: T, B, C
        padding_mask: T, B (True for padded positions)
        """
        if padding_mask is None:
            return encoder_out.mean(dim=0)
        else:
            dtype = encoder_out.dtype
            encoder_out[padding_mask, :] = 0.0
            lengths = (~padding_mask).float().sum(dim=0)
            out = encoder_out.float().sum(dim=0) / lengths.unsqueeze(-1)
            return out.to(dtype)


class SelfAttnPooler(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, encoder_out, padding_mask):
        """
        encoder_out: T, B, C
        padding_mask: T, B (True for padded positions)
        """
        dtype = encoder_out.dtype
        attn_weights = self.proj(encoder_out).squeeze(-1).float()
        if padding_mask is not None:
            attn_weights[padding_mask] = float("-inf")
        attn_weights = attn_weights.softmax(dim=0)
        out = torch.einsum("tb,tbc->bc", attn_weights.float(), encoder_out.float())
        return out.to(dtype)
