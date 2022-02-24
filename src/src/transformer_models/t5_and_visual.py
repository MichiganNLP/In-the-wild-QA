from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from overrides import overrides
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Stack


def _combine_attention_masks(text_attention_mask: torch.Tensor | None = None,
                             visual_attention_mask: torch.Tensor | None = None) -> torch.Tensor | None:
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
                attention_mask: torch.Tensor | None = None, visual_attention_mask: torch.Tensor | None = None,
                **kwargs) -> BaseModelOutputWithPastAndCrossAttentions | tuple[torch.Tensor, ...]:
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
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, visual: torch.Tensor | None = None,
                                      visual_attention_mask: torch.Tensor | None = None,
                                      **kwargs) -> Mapping[str, Any]:
        output = super().prepare_inputs_for_generation(input_ids, **kwargs)  # noqa
        output["visual"] = visual
        output["visual_attention_mask"] = visual_attention_mask
        return output

    @overrides(check_signature=False)
    def forward(self, masked_caption_ids: torch.Tensor | None = None, visual: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, visual_attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None, **kwargs) -> Seq2SeqLMOutput | tuple[torch.Tensor, ...]:
        if "encoder_outputs" not in kwargs:
            kwargs["encoder_outputs"] = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                                                     visual_attention_mask=visual_attention_mask, **kwargs)

        # The attention mask used in the encoder (the combined mask) is actually necessary for the decoder even when
        # providing "encoder_outputs". We could compute it once in the encoder, return it as one of its fields and use
        # it here. However, in `T5FillerModel.generative_step` we input the encoder outputs but without the mask
        # since it's constructed from the `generate` output which in turn only returns certain fields (not the mask).
        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)
        return super().forward(attention_mask=attention_mask, labels=labels, **kwargs)  # noqa


class T5AndVisualEvidence(T5EncoderModel):  # noqa
    def __init__(self, config: T5Config, visual_size: int) -> None:
        super().__init__(config)
        self.encoder = TextVisualEncoder(self.encoder, visual_size)
        self.start_end = nn.Linear(self.config.d_model, 2)

    @overrides(check_signature=False)
    def forward(self, masked_caption_ids: torch.Tensor | None = None, visual: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, visual_attention_mask: torch.Tensor | None = None,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                               visual_attention_mask=visual_attention_mask, **kwargs)

        visual_start = masked_caption_ids.shape[1]
        visual_hidden = outputs.last_hidden_state[:, visual_start:, :]

        start_end = self.start_end(visual_hidden)

        return start_end[..., 0], start_end[..., 1]

    def predict(self, masked_caption_ids: torch.Tensor | None = None, visual: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                visual_attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self(masked_caption_ids, visual, attention_mask, visual_attention_mask)
        return start.log_softmax(dim=-1), end.log_softmax(dim=-1)


class T5AndVisualEvidenceIO(T5EncoderModel):  # noqa
    def __init__(self, config: T5Config, visual_size: int) -> None:
        super().__init__(config)
        self.encoder = TextVisualEncoder(self.encoder, visual_size)
        self.linear = nn.Linear(self.config.d_model, 1)

    @overrides(check_signature=False)
    def forward(self, masked_caption_ids: torch.Tensor | None = None, visual: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, visual_attention_mask: torch.Tensor | None = None,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                               visual_attention_mask=visual_attention_mask, **kwargs)
        visual_start = masked_caption_ids.shape[1]
        visual_hidden = outputs.last_hidden_state[:, visual_start:, :]
        score = self.linear(visual_hidden)
        return torch.sigmoid(score[..., 0])  # calculate the probability that the frame is "I"

    def predict(self, masked_caption_ids: torch.Tensor | None = None, visual: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                visual_attention_mask: torch.Tensor | None = None) -> Mapping[int, Iterable[tuple[int, int, float]]]:
        prob_in = self(masked_caption_ids, visual, attention_mask, visual_attention_mask)
        in_index = (prob_in >= 0.5).nonzero(as_tuple=False).cpu()  # move variables to cpu to append to list
        results = defaultdict(list)
        # get the position for each frame that is within the evidence
        for batch_idx, frame_idx in in_index:
            batch_idx, frame_idx = int(batch_idx), int(frame_idx)
            if results[batch_idx]:
                last_frame_idx = results[batch_idx][-1][1]
                if last_frame_idx + 1 == frame_idx:
                    results[batch_idx][-1][1] = frame_idx
                else:
                    results[batch_idx].append([frame_idx, frame_idx])
            else:
                results[batch_idx].append([frame_idx, frame_idx])

        results_with_score = defaultdict(list)
        # get the score of the evidence
        for b, start_ends in results.items():
            for start_idx, end_idx in start_ends:
                score = torch.mean(prob_in[b][start_idx: end_idx + 1]).cpu()
                results_with_score[b].append([start_idx, end_idx, float(score)])

        # sort the evidence based on the score
        for b in results_with_score:
            results_with_score[b].sort(key=lambda x: x[-1], reverse=True)
        return results_with_score


class T5MultiTask(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, visual_size: int) -> None:
        super().__init__(config)
        self.encoder = TextVisualEncoder(self.encoder, visual_size)  # noqa
        self.start_end = nn.Linear(self.config.d_model, 2)  # noqa

    @overrides(check_signature=False)
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, visual: torch.Tensor | None = None,
                                      visual_attention_mask: torch.Tensor | None = None, **kwargs) -> Mapping[str, Any]:
        output = super().prepare_inputs_for_generation(input_ids, **kwargs)  # noqa
        output["visual"] = visual
        output["visual_attention_mask"] = visual_attention_mask
        return output

    @overrides(check_signature=False)
    def forward(self, masked_caption_ids: torch.Tensor | None = None, visual: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, visual_attention_mask: torch.Tensor | None = None,
                labels: torch.Tensor | None = None, **kwargs) -> Seq2SeqLMOutput:
        start = end = None
        if "encoder_outputs" not in kwargs:  # Only here when doing multitask training
            start, end, encoder_outputs = self._evidence_forward(masked_caption_ids, visual=visual,
                                                                 attention_mask=attention_mask,
                                                                 visual_attention_mask=visual_attention_mask, **kwargs)
            kwargs["encoder_outputs"] = encoder_outputs

        # The attention mask used in the encoder (the combined mask) is actually necessary for the decoder even when
        # providing "encoder_outputs". We could compute it once in the encoder, return it as one of its fields and use
        # it here. However, in `T5FillerModel.generative_step` we input the encoder outputs but without the mask
        # since it's constructed from the `generate` output which in turn only returns certain fields (not the mask).
        attention_mask = _combine_attention_masks(attention_mask, visual_attention_mask)
        outputs = super().forward(attention_mask=attention_mask, labels=labels, **kwargs)  # noqa

        if start is not None:
            setattr(outputs, "start", start)

        if end is not None:
            setattr(outputs, "end", end)

        return outputs

    def _evidence_forward(self, masked_caption_ids: torch.Tensor | None = None, visual: torch.Tensor | None = None,
                          attention_mask: torch.Tensor | None = None, visual_attention_mask: torch.Tensor | None = None,
                          **kwargs) -> tuple[torch.Tensor, torch.Tensor, Seq2SeqLMOutput]:
        encoder_outputs = self.encoder(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                                       visual_attention_mask=visual_attention_mask, **kwargs)
        visual_start = masked_caption_ids.shape[1]
        visual_hidden = encoder_outputs.last_hidden_state[:, visual_start:, :]

        start_end = self.start_end(visual_hidden)
        return start_end[..., 0], start_end[..., 1], encoder_outputs

    def predict(self, masked_caption_ids: torch.Tensor | None = None, visual: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, visual_attention_mask: torch.Tensor | None = None,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        start, end, _ = self._evidence_forward(masked_caption_ids, visual=visual, attention_mask=attention_mask,
                                               visual_attention_mask=visual_attention_mask, **kwargs)
        return start.log_softmax(dim=-1), end.log_softmax(dim=-1)
