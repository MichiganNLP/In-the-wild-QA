from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
from overrides import overrides
from torch import nn
from transformers import CLIPConfig, CLIPModel, T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from src.transformer_models.t5_and_visual import _combine_attention_masks


class ClipWithDecoder(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, pretrained_clip_ckpt_path: str, max_seq: int) -> None:
        super().__init__(config)
        clip_config = CLIPConfig.from_pretrained(pretrained_clip_ckpt_path)
        self.clip_model = CLIPModel.from_pretrained(pretrained_clip_ckpt_path)
        setattr(self.clip_model.text_model.embeddings, "position_ids", torch.arange(max_seq).expand((1, -1)))
        setattr(self.clip_model.text_model.embeddings, "position_embedding",
                nn.Embedding(max_seq, clip_config.text_config.hidden_size))
        self.text_enc = self.clip_model.text_model
        self.visual_enc = self.clip_model.vision_model
        # TODO: change later - santi: what thing??
        self.emb_temporal = nn.Parameter(0.02 * torch.randn(500, clip_config.text_config.hidden_size))
        self.linear = nn.Linear(clip_config.text_config.hidden_size, 768)

    @overrides(check_signature=False)
    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.Tensor,
                                                       model_kwargs: MutableMapping[str, Any]) -> Mapping[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            text_output = self.text_enc(input_ids=input_ids, attention_mask=model_kwargs["attention_mask"])

            # TODO: can we use the pooled output?
            visual_outputs = torch.stack([self.clip_model.visual_projection(self.visual_enc(pixel_values=v)[1])
                                          for v in model_kwargs["visual"]])

            _B, _T, _D = visual_outputs.shape
            # NOTE: here with batch size == 1
            visual_outputs += self.emb_temporal.expand(_B, -1, -1)[:, :_T, :]
            # NOTE: here recovering the batch size of 1
            # Add extra 1 dimension to match dimension after encoder_outputs[0]
            model_kwargs["encoder_outputs"] = BaseModelOutput(
                last_hidden_state=self.linear(torch.cat([text_output[0], visual_outputs], dim=1)))

        return model_kwargs

    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor, **kwargs) -> Seq2SeqLMOutput:
        kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs=kwargs)
        attention_mask = _combine_attention_masks(kwargs["attention_mask"], kwargs["visual_attention_mask"])
        return super().forward(attention_mask=attention_mask, labels=kwargs["labels"], **kwargs)
