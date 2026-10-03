"""Hugging Face model classes for Nuri (ARCHITECTURE.md §15).

Load a released model with trust_remote_code=True:

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    model = AutoModelForMaskedLM.from_pretrained("path/or/repo", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("path/or/repo")

Every class wraps one NuriModel under the attribute `model`, so the weight
names in model.safetensors start with "model.". Besides the usual forward,
all classes offer:

- encode(input_ids, ...)          unit-length sentence embeddings
- generate_refine(input_ids, ...) masked-refinement text generation

Nuri has no autoregressive decoder, so the standard generate() does not apply.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput

from .configuration_nuri import NuriHFConfig
from .core import RecurrentReasoningCore
from .generation import GenerationResult, generate
from .layers import RMSNorm
from .model import NuriModel
from .moe import MoEFFN


class NuriPreTrainedModel(PreTrainedModel):
    config_class = NuriHFConfig
    base_model_prefix = "model"

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Same initialization as NuriModel; used only for weights missing from a checkpoint."""
        std = self.config.nuri.get("initializer_range", 0.02)
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, std=std)
        elif isinstance(module, MoEFFN):
            for p in (module.w_gate, module.w_up, module.w_down):
                torch.nn.init.normal_(p, std=std)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RecurrentReasoningCore):
            torch.nn.init.normal_(module.slots, std=std)
            gate = self.config.nuri.get("gate_init", 0.1)
            torch.nn.init.constant_(module.gate_logits, math.log(gate / (1 - gate)))

    def _with_moe_aux(self, loss):
        """Collect this forward's MoE balance/z losses (training mode only) and add them to `loss`.

        Always collecting also clears the records, so repeated training
        forwards never pile them up.
        """
        aux = self.model.pop_moe_aux()["loss"]
        if loss is None or aux is None:
            return loss
        return loss + aux

    def encode(self, input_ids, attention_mask=None, mode: str = "fast", dim: int | None = None,
               steps: int | None = None) -> torch.Tensor:
        """Sentence embeddings. mode: "fast" (no reasoning) or "reasoning" (T = steps)."""
        return self.model.embed(input_ids, attention_mask, mode=mode, steps=steps, dim=dim)

    def generate_refine(self, input_ids, attention_mask=None, T: int | None = None, S: int | None = None,
                        max_len: int | None = None, canvas_len=None, temperature: float = 0.0,
                        top_k: int | None = None, generator: torch.Generator | None = None) -> GenerationResult:
        """T = reasoning steps, S = refinement rounds. Base models need canvas_len.

        temperature 0 is greedy; above 0 samples (optionally top_k) with `generator`.
        """
        return generate(self.model, input_ids, attention_mask, steps=T, rounds=S, canvas_len=canvas_len,
                        max_len=max_len, temperature=temperature, top_k=top_k, generator=generator)


class NuriHFModel(NuriPreTrainedModel):
    """AutoModel: returns the input memory X as last_hidden_state."""

    def __init__(self, config: NuriHFConfig):
        super().__init__(config)
        self.model = NuriModel(config.to_nuri_config())
        self.post_init()

    def forward(self, input_ids, attention_mask=None, **kwargs) -> BaseModelOutput:
        x, _ = self.model.encode(input_ids, attention_mask)
        self._with_moe_aux(None)  # no loss here; just clear the records
        return BaseModelOutput(last_hidden_state=x)


class NuriForMaskedLM(NuriPreTrainedModel):
    """AutoModelForMaskedLM: token logits straight from X through the tied vocabulary head."""

    def __init__(self, config: NuriHFConfig):
        super().__init__(config)
        self.model = NuriModel(config.to_nuri_config())
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs) -> MaskedLMOutput:
        x, _ = self.model.encode(input_ids, attention_mask)
        logits = self.model.vocab_logits(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), ignore_index=-100)
        return MaskedLMOutput(loss=self._with_moe_aux(loss), logits=logits)


class NuriForSequenceClassification(NuriPreTrainedModel):
    """AutoModelForSequenceClassification: a head on concat(mean X, mean R).

    The head is new for every downstream task, so from_pretrained initializes
    it and reports it as missing; that is expected.
    """

    def __init__(self, config: NuriHFConfig):
        super().__init__(config)
        self.model = NuriModel(config.to_nuri_config(num_labels=config.num_labels))
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, reasoning_steps: int | None = None,
                **kwargs) -> SequenceClassifierOutput:
        x, valid = self.model.encode(input_ids, attention_mask)
        u, r = self.model.pool(x, valid, self.model.reason(x, valid, reasoning_steps))
        logits = self.model.task_logits(u, r)
        loss = F.cross_entropy(logits.float(), labels) if labels is not None else None
        return SequenceClassifierOutput(loss=self._with_moe_aux(loss), logits=logits)


AUTO_MAP = {
    "AutoConfig": "configuration_nuri.NuriHFConfig",
    "AutoModel": "modeling_nuri.NuriHFModel",
    "AutoModelForMaskedLM": "modeling_nuri.NuriForMaskedLM",
    "AutoModelForSequenceClassification": "modeling_nuri.NuriForSequenceClassification",
}
