from __future__ import annotations

from typing import Callable, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from deepdct.interpretability.hooks import resolve_module, tensor_from_output


ModelOutput = Union[Tensor, Mapping[str, Tensor]]
TargetFunction = Callable[[ModelOutput], Tensor]


class RegressionGradCAM:
    """
    Grad-CAM for a regression network.

    The target function must return a scalar tensor derived from the model output.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
    ) -> None:
        self.model = model
        self.target_layer_name = target_layer
        self.target_layer = resolve_module(model, target_layer)

        self.activation: Optional[Tensor] = None
        self.gradient: Optional[Tensor] = None

        self._forward_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )

        self._backward_handle = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: Tuple[object, ...],
        output: object,
    ) -> None:
        self.activation = tensor_from_output(output)

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[Tensor], ...],
        grad_output: Tuple[Optional[Tensor], ...],
    ) -> None:
        for gradient in grad_output:
            if isinstance(gradient, Tensor):
                self.gradient = gradient
                return

    def generate(
        self,
        model_inputs: Mapping[str, object],
        target_function: TargetFunction,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, ModelOutput]:
        """
        Returns:
            cam: [B, 1, H, W], normalized to [0, 1]
            outputs: original model outputs
        """
        self.activation = None
        self.gradient = None

        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            outputs = self.model(**model_inputs)
            target_scalar = target_function(outputs)

            if target_scalar.ndim != 0:
                target_scalar = target_scalar.sum()

            target_scalar.backward(retain_graph=False)

        if self.activation is None:
            raise RuntimeError(
                f"No activation was captured from '{self.target_layer_name}'."
            )

        if self.gradient is None:
            raise RuntimeError(
                f"No gradient was captured from '{self.target_layer_name}'. "
                "Check that this layer contributes to the selected output."
            )

        activation = self.activation
        gradient = self.gradient

        if activation.ndim != 4 or gradient.ndim != 4:
            raise ValueError(
                "Grad-CAM currently expects BCHW feature tensors. "
                f"Activation shape: {tuple(activation.shape)}, "
                f"gradient shape: {tuple(gradient.shape)}."
            )

        # Global-average-pool gradients over spatial dimensions.
        weights = gradient.mean(dim=(2, 3), keepdim=True)

        # Weighted feature-map combination.
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        if output_size is not None:
            cam = F.interpolate(
                cam,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)

        cam = (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-8)

        return cam.detach().cpu(), outputs

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "RegressionGradCAM":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()