from typing import Any, Dict, Mapping, Optional, Union

import torch
from overrides import overrides
from torch import nn
from transformers import CLIPConfig, CLIPModel, T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from src.transformer_models.t5_and_visual import _combine_attention_masks


class CLIPWithDecoder(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, pretrained_clip_ckpt_path: str, max_seq: int) -> None:
        super().__init__(config)
        clip_config = CLIPConfig.from_pretrained(pretrained_clip_ckpt_path)
        self.clip_model = CLIPModel.from_pretrained(pretrained_clip_ckpt_path)
        setattr(self.clip_model.text_model.embeddings, "position_ids", torch.arange(max_seq).expand((1, -1)))
        setattr(self.clip_model.text_model.embeddings, "position_embedding",
                nn.Embedding(max_seq, clip_config.text_config.hidden_size))
        self.text_enc = self.clip_model.text_model
        self.visual_enc = self.clip_model.vision_model
        # TODO: change later
        self.emb_temporal = torch.nn.Parameter(0.02 * torch.randn(500, clip_config.text_config.hidden_size))
        self.linear = nn.Linear(clip_config.text_config.hidden_size, 768)

    @overrides(check_signature=False)
    def _prepare_encoder_decoder_kwargs_for_generation(
            self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        txt = input_ids
        attention_mask = model_kwargs["attention_mask"]
        visual = model_kwargs["visual"]
        visual_attention_mask = model_kwargs["visual_attention_mask"]
        model_kwargs = self._encoder_forward(txt, attention_mask, visual, visual_attention_mask, model_kwargs)
        return model_kwargs

    def _encoder_forward(self, txt, attention_mask, visual, visual_attention_mask, kwargs):
        if "encoder_outputs" not in kwargs:
            text_output = self.text_enc(input_ids=txt, attention_mask=attention_mask)
            visual_outputs = []
            for v in visual:
                # TODO: can we use the pooled output?
                visual_outputs.append(self.clip_model.visual_projection(self.visual_enc(pixel_values=v)[1]))

            visual_outputs = torch.stack(visual_outputs, dim=0)
            _B, _T, _D = visual_outputs.shape
            # NOTE: here with batch size == 1
            visual_outputs += self.emb_temporal.expand(_B, -1, -1)[:, :_T, :]
            # NOTE: here recovering the batch size of 1
            # Add extra 1 dimension to match dimension after encoder_outputs[0]
            kwargs["encoder_outputs"] = BaseModelOutput(
                last_hidden_state=self.linear(torch.cat([text_output[0], visual_outputs], dim=1))
            )
        return kwargs

    @overrides(check_signature=False)
    def forward(self, txt: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
                visual: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, **kwargs) -> Union[Seq2SeqLMOutput, tuple[torch.Tensor, ...]]:
        kwargs = self._encoder_forward(txt, attention_mask, visual, visual_attention_mask, kwargs)
        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)
        return super().forward(attention_mask=attention_mask, labels=labels, **kwargs)  # noqa

    # TODO: change the code here
    @overrides(check_signature=False)
    def prepare_inputs_for_generation(self, txt: torch.Tensor, visual: Optional[torch.Tensor] = None,
                                      visual_attention_mask: Optional[torch.Tensor] = None,
                                      **kwargs) -> Mapping[str, Any]:
        output = super().prepare_inputs_for_generation(txt, **kwargs)  # noqa
        output["visual"] = visual
        output["visual_attention_mask"] = visual_attention_mask
        return output
