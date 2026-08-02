from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


def resolve_module(model: nn.Module, module_path: str) -> nn.Module:
    """
    Resolve a dotted module path.

    Examples:
        resolve_module(model, "aresunet.encoder.stage1")
        resolve_module(model, "model_r.encoder.blocks.2")
    """
    current: nn.Module = model

    for part in module_path.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            if not hasattr(current, part):
                available = ", ".join(name for name, _ in current.named_children())
                raise AttributeError(
                    f"Module '{current.__class__.__name__}' has no child "
                    f"named '{part}'. Available children: [{available}]"
                )
            current = getattr(current, part)

        if not isinstance(current, nn.Module):
            raise TypeError(
                f"Resolved object at '{part}' is not an nn.Module: "
                f"{type(current).__name__}"
            )

    return current


def tensor_from_output(output: object) -> Tensor:
    """
    Extract the first tensor from common PyTorch module-output structures.
    """
    if isinstance(output, Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, Tensor):
                return item

    if isinstance(output, Mapping):
        for item in output.values():
            if isinstance(item, Tensor):
                return item

    raise TypeError(
        "Hooked module did not return a Tensor or a supported tensor container."
    )


class FeatureMapCollector:
    """
    Collect outputs from selected layers using forward hooks.

    Example:
        collector = FeatureMapCollector(
            model,
            ["aresunet.encoder.stage1", "aresunet.decoder.stage3"],
        )

        with collector:
            outputs = model(...)

        maps = collector.activations
    """

    def __init__(
        self,
        model: nn.Module,
        module_paths: Iterable[str],
        detach: bool = True,
        move_to_cpu: bool = True,
    ) -> None:
        self.model = model
        self.module_paths = list(module_paths)
        self.detach = detach
        self.move_to_cpu = move_to_cpu

        self.activations: "OrderedDict[str, Tensor]" = OrderedDict()
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str):
        def hook(
            module: nn.Module,
            inputs: Tuple[object, ...],
            output: object,
        ) -> None:
            tensor = tensor_from_output(output)

            if self.detach:
                tensor = tensor.detach()

            if self.move_to_cpu:
                tensor = tensor.cpu()

            self.activations[name] = tensor

        return hook

    def register(self) -> None:
        if self._handles:
            return

        for module_path in self.module_paths:
            module = resolve_module(self.model, module_path)
            handle = module.register_forward_hook(self._make_hook(module_path))
            self._handles.append(handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()

        self._handles.clear()

    def clear(self) -> None:
        self.activations.clear()

    def __enter__(self) -> "FeatureMapCollector":
        self.clear()
        self.register()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.remove()


def collect_internal_attention_maps(
    model: nn.Module,
) -> Dict[str, Tensor]:
    """
    Collect tensors stored by attention modules as `last_attention_map`.
    """
    maps: Dict[str, Tensor] = {}

    for name, module in model.named_modules():
        attention = getattr(module, "last_attention_map", None)

        if isinstance(attention, Tensor):
            maps[name] = attention.detach().cpu()

    return maps