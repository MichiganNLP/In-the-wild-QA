from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
from overrides import overrides
from torch import nn
from transformers import AutoModel, T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from src.transformer_models.t5_and_visual import _combine_attention_masks


class ClipWithDecoder(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, pretrained_clip_ckpt_path: str, max_seq: int,
                 max_visual_features: int = 500) -> None:
        super().__init__(config)

        self.model = AutoModel.from_pretrained(pretrained_clip_ckpt_path)

        text_or_visual_encoded_size = self.model.config.text_config.hidden_size
        self.embed_temporal = nn.Parameter(0.02 * torch.randn(max_visual_features, text_or_visual_encoded_size))
        self.linear = nn.Linear(text_or_visual_encoded_size, config.d_model)

    @overrides(check_signature=False)
    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.Tensor,
                                                       model_kwargs: MutableMapping[str, Any],
                                                       model_input_name: None = None) -> Mapping[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            assert model_input_name is None

            text_mask = model_kwargs["attention_mask"]

            encoded_text = self.model.get_text_features(input_ids=input_ids, attention_mask=text_mask)

            visual = model_kwargs["visual"]

            batch_size, frame_count = visual.shape[:2]
            visual = visual.view(batch_size * frame_count, *visual.shape[2:])

            encoded_visual = (self.model.get_image_features(visual).view(batch_size, frame_count, -1)
                              + self.embed_temporal.expand(batch_size, -1, -1)[:, :frame_count])

            encoded = self.linear(torch.cat([encoded_text.unsqueeze(dim=1), encoded_visual], dim=1))
            model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=encoded)

            model_kwargs["attention_mask"] = _combine_attention_masks(text_mask[:, :1],
                                                                      model_kwargs["visual_attention_mask"])

            del model_kwargs["visual"]
            del model_kwargs["visual_attention_mask"]

        return model_kwargs

    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor | None = None, **kwargs) -> Seq2SeqLMOutput:
        # `input_ids` can be None because `encoder_outputs` can be present.
        return super().forward(**self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs=kwargs))
