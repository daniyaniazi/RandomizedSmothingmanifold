"""NER transformer model for token classification.

The model is a plain predictor — no smoothing logic inside.
Smoothing is applied externally before calling forward().

Three entry points supported by the pipeline:
  - input_ids          → standard forward, no smoothing
  - inputs_embeds      → smoothing applied at embedding level (Study A)
  - layer_noise_fn     → smoothing injected at a specific transformer layer (Study B)

hidden_states and attentions are always returned for layer-wise analysis (Study B/C).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class NERForwardOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]
    hidden_states: Optional[tuple] = field(default=None, repr=False)
    attentions: Optional[tuple] = field(default=None, repr=False)


class TransformerNER(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        dropout: float = 0.1,
    ):
        super().__init__()
        cfg = AutoConfig.from_pretrained(
            encoder_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        self.encoder = AutoModel.from_pretrained(encoder_name, config=cfg)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def num_layers(self) -> int:
        """Return number of transformer layers in the encoder."""
        for attr in ("layer", "layers"):
            layers = (
                getattr(getattr(self.encoder, "encoder", None), attr, None)
                or getattr(getattr(self.encoder, "transformer", None), attr, None)
            )
            if layers is not None:
                return len(layers)
        return 0

    def _layer_list(self):
        for attr in ("layer", "layers"):
            layers = (
                getattr(getattr(self.encoder, "encoder", None), attr, None)
                or getattr(getattr(self.encoder, "transformer", None), attr, None)
            )
            if layers is not None:
                return layers
        return None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        layer_noise_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        return_attentions: bool = False,
    ) -> NERForwardOutput:
        """
        Parameters
        ----------
        layer_noise_fn : optional callable(hidden: Tensor, layer_idx: int) -> Tensor
            If provided, is called after every transformer layer.
            The returned tensor replaces that layer's hidden states.
            Use this for Study (layer-wise manifold smoothing).
        return_attentions : bool
            If True, attention weights for all layers are returned in output.
        """
        if attention_mask is None:
            raise ValueError("attention_mask must be provided.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        hook_handles: List = []
        if layer_noise_fn is not None:
            layers = self._layer_list()
            if layers is not None:
                for idx, layer_module in enumerate(layers):
                    i = idx

                    def _make_hook(layer_idx):
                        def _hook(_, __, output):
                            h = output[0] if isinstance(output, tuple) else output
                            h = h + layer_noise_fn(h, layer_idx)
                            return (h,) + output[1:] if isinstance(output, tuple) else h
                        return _hook

                    hook_handles.append(
                        layer_module.register_forward_hook(_make_hook(i))
                    )

        enc = self.encoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=return_attentions,
            return_dict=True,
        )

        for h in hook_handles:
            h.remove()

        logits = self.classifier(self.dropout(enc.last_hidden_state))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return NERForwardOutput(
            logits=logits,
            loss=loss,
            hidden_states=enc.hidden_states,
            attentions=enc.attentions if return_attentions else None,
        )
