from typing import Any, Mapping, Optional, Union

import torch
from overrides import overrides
from torch import nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack


def _combine_attention_masks(text_attention_mask: Optional[torch.Tensor] = None,
                             visual_attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    if text_attention_mask is not None and visual_attention_mask is not None:
        text_batch_size = text_attention_mask.shape[0]
        visual_batch_size = visual_attention_mask.shape[0]
        beam_size = text_batch_size // visual_batch_size
        if beam_size > 1:
            visual_attention_mask = visual_attention_mask.repeat(beam_size, 1)
        return torch.cat([text_attention_mask, visual_attention_mask], dim=1)
    else:
        assert text_attention_mask is None and visual_attention_mask is None, \
            "Can't set the text or visual attention mask as one is empty and the other one isn't."
        return None


class TextVisualEncoder(T5Stack):  # noqa
    def __init__(self, t5stack: T5Stack, visual_size: int) -> None:
        super().__init__(t5stack.config, t5stack.embed_tokens)
        self.embed_video = nn.Linear(visual_size, self.embed_tokens.embedding_dim)

    @overrides(check_signature=False)
    def forward(self, text_token_ids: torch.Tensor, visual: torch.Tensor,  # noqa
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Union[BaseModelOutputWithPastAndCrossAttentions, tuple[torch.Tensor, ...]]:
        text_embedding = self.embed_tokens(text_token_ids)
        visual_embedding = self.embed_video(visual)
        embedding = torch.cat([text_embedding, visual_embedding], dim=1)
        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)

        return super().forward(inputs_embeds=embedding, attention_mask=attention_mask, **kwargs)


class T5AndVisual(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, visual_size: int) -> None:
        super().__init__(config)
        self.encoder = TextVisualEncoder(self.encoder, visual_size)  # noqa

    @overrides(check_signature=False)
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, visual: Optional[torch.Tensor] = None,
                                      visual_attention_mask: Optional[torch.Tensor] = None,
                                      **kwargs) -> Mapping[str, Any]:
        output = super().prepare_inputs_for_generation(input_ids, **kwargs)  # noqa
        output["visual"] = visual
        output["visual_attention_mask"] = visual_attention_mask
        return output

    @overrides(check_signature=False)
    def forward(self, masked_caption_ids: Optional[torch.Tensor] = None, visual: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, visual_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, **kwargs) -> Union[Seq2SeqLMOutput, tuple[torch.Tensor, ...]]:
        if "encoder_outputs" not in kwargs:
            kwargs["encoder_outputs"] = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                                                     visual_attention_mask=visual_attention_mask, **kwargs)

        # The attention mask used in the encoder (the combined mask) is actually necessary for the decoder even when
        # providing "encoder_outputs". We could compute it once in the encoder, return it as one of its fields and use
        # it here. However, in `T5FillerModel.generative_step` we input the encoder outputs but without the mask
        # since it's constructed from the `generate` output which in turn only returns certain fields (not the mask).
        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)

        return super().forward(attention_mask=attention_mask, labels=labels, **kwargs)  # noqa
