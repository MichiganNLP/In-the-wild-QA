import torch
from transformers import PretrainedConfig


def compute_answer_probs(logits: torch.Tensor, answer_ids: torch.Tensor, model_config: PretrainedConfig,
                         ignore_eos_token: bool = False) -> torch.Tensor:
    """Computes the probability of the given answer token using the logits.

    `logits` has shape (N, L, V) and dtype float.
    `answer_ids` has shape (N, L) and dtype int.

    Returned tensor has shape (N, L) and dtype float.
    """
    if model_config.decoder_start_token_id is not None \
            and (answer_ids[:, 0] == model_config.decoder_start_token_id).all():  # noqa
        answer_ids = answer_ids[:, 1:]

    N, L = answer_ids.shape

    probs = logits.softmax(dim=-1)
    answer_probs = probs[torch.arange(N)[:, None], torch.arange(L)[None], answer_ids]

    if model_config.pad_token_id is not None:
        answer_probs[answer_ids == model_config.pad_token_id] = 1

    if ignore_eos_token and model_config.eos_token_id is not None:
        answer_probs[answer_ids == model_config.eos_token_id] = 1

    return answer_probs


def compute_answer_prob(answer_probs: torch.Tensor) -> torch.Tensor:
    """Computes the joint probability of the given answer.

    `answer_probs` has shape (N, L) and dtype float.

    Returned tensor has shape (N,) and dtype float.
    """
    # There should be just a few factors, so the product should be numerically stable.
    return answer_probs.prod(dim=-1)
