# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

from argparse import Namespace
import contextlib
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.tasks import FairseqTask

from fairseq.models.hubert import HubertAsrConfig, HubertCtc, HubertEncoder

@dataclass
class SpeechLMCtcNerConfig(HubertAsrConfig):
    autoregressive: bool = field(
        default=False, metadata={"help": "required by AudioFinetuningTask"},
    )
    freeze_layers: int = field(
        default=0, metadata={"help": "dont finetune this many bottom layers in speechlm text encoder"}
    )


@register_model("speechlm_ctc_ner", dataclass=SpeechLMCtcNerConfig)
class SpeechLMCtcNer(HubertCtc):
    def __init__(self, cfg: SpeechLMCtcNerConfig, w2v_encoder: BaseFairseqModel):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(cls, cfg: SpeechLMCtcNerConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = SpeechLMEncoder(cfg, task)
        return cls(cfg, w2v_encoder)

    def forward(self, **kwargs):
        assert "source" in kwargs or "src_tokens" in kwargs
        if "source" in kwargs and kwargs["source"].dtype!=torch.int64:
            x = self.w2v_encoder(**kwargs)
        else:
            x = self.w2v_encoder.forward_text(**kwargs)
        return x


class SpeechLMEncoder(FairseqEncoder):
    """
    Modified from fairseq.models.hubert.hubert_asr.HubertEncoder
    1. make it compatible with fairseq speech_to_text task
    2. make it compatible with encoder-decoder model
    """
    def __init__(self, cfg: HubertAsrConfig, task):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "freeze_layers": cfg.freeze_layers
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        pretrain_task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            pretrain_task.load_state_dict(state["task_state"])
        else:
            pretrain_task.load_state_dict(task.state_dict())

        model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        super().__init__(pretrain_task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0
        
        if (task.target_dictionary is not None) and (
            hasattr(self.w2v_model, "unit_encoder_ctc_head")
        ):
            self.proj = nn.Linear(d, len(task.target_dictionary))
            self.conv_ctc_proj = True
        else:
            self.conv_ctc_proj = False

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):
           
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)
            
            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        results = {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

        if self.conv_ctc_proj:
            padding_mask = self.w2v_model.downsample_ctc_padding_mask(results["padding_mask"])
            results["encoder_padding_mask"] = padding_mask
            results["padding_mask"] = padding_mask
        return results

    def forward_text(self, source=None, src_tokens=None, tbc=True, **kwargs):
        assert source is not None or src_tokens is not None
        source = source if source is not None else src_tokens
        
        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            encoder_out = self.w2v_model.unit_encoder(
                source,
                freeze_layers=self.w2v_model.freeze_layers,
            )

        encoder_out["encoder_out"] = self.final_dropout(encoder_out["encoder_out"][0])

        if self.proj:
            encoder_out["encoder_out"] = self.proj(encoder_out["encoder_out"])

        results = {
            "encoder_out": encoder_out["encoder_out"],  # T x B x C
            "encoder_padding_mask": encoder_out["encoder_padding_mask"][0],  # B x T
            "padding_mask": encoder_out["encoder_padding_mask"][0],
        }
        
        if self.conv_ctc_proj:
            padding_mask = self.w2v_model.downsample_ctc_padding_mask(results["padding_mask"])
            results["encoder_padding_mask"] = padding_mask
            results["padding_mask"] = padding_mask
        return results

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = [
                x.index_select(1, new_order) for x in encoder_out["encoder_out"]
            ]
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = [
                x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]
            ]
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict
