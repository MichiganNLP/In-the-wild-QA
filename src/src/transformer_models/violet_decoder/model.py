from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from overrides import overrides
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.transformer_models.t5_and_visual import _combine_attention_masks
from src.transformer_models.violet_decoder.pytorch_violet.model import VIOLET_Base


class VioletWithDecoder(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, pretrained_violet_ckpt_path: str) -> None:
        super().__init__(config)
        self.encoder = VIOLET_Base()
        self.encoder.load_ckpt(pretrained_violet_ckpt_path)

    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                visual: torch.Tensor | None = None, visual_attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None, **kwargs) -> Seq2SeqLMOutput | tuple[torch.Tensor, ...]:
        if "encoder_outputs" not in kwargs:
            (_B, _T, _, _H, _W), (_, _X) = visual.shape, input_ids.shape
            _h, _w = _H // 32, _W // 32

            feat_img, _, feat_txt, mask_txt = self.encoder.go_feat(visual, input_ids, attention_mask)
            # build image mask ourselves
            mask_img = torch.repeat_interleave(visual_attention_mask, (1 + _h * _w), dim=1)
            encoder_output, encoder_attention_output = self.encoder.go_cross(feat_img, mask_img, feat_txt, mask_txt)
            kwargs["encoder_outputs"] = torch.unsqueeze(encoder_output, 0)

        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)
        return super().forward(attention_mask=attention_mask, labels=labels, **kwargs)  # noqa

    # TODO: change the code here
    @overrides(check_signature=False)
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, visual: torch.Tensor | None = None,
                                      attention_mask: torch.Tensor | None = None, **kwargs) -> Mapping[str, Any]:
        output = super().prepare_inputs_for_generation(input_ids, attention_mask=attention_mask, **kwargs)  # noqa
        output["img"] = visual
        return output
