import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCam(nn.Module):
    def __init__(self, model, target_layer):
        super().__init__()

        self.model = model
        self.target_layer = target_layer

        self.forward_result = None
        self.backward_result = None

        self.forward_handle = None
        self.backward_handle = None

        self.register_hooks()

    def register_hooks(self):
        self.forward_handle = self.target_layer.register_forward_hook(
            self.forward_hook
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward(self, input, target_index=None):
        """
        input shape: [1, C, H, W]
        return: Grad-CAM heatmap, shape [H, W]
        """
        assert input.size(0) == 1, "Grad-CAM expects batch size 1."

        self.model.eval()

        self.forward_result = None
        self.backward_result = None

        outs = self.model(input)

        if outs.dim() == 2:
            outs = outs[0]

        if target_index is None:
            target_index = outs.argmax().item()

        self.model.zero_grad(set_to_none=True)

        target_score = outs[target_index]
        target_score.backward(retain_graph=True)

        if self.forward_result is None or self.backward_result is None:
            raise RuntimeError("Forward or backward hook did not capture results.")

        gradients = self.backward_result
        activations = self.forward_result

        weights = torch.mean(
            gradients,
            dim=(1, 2),
            keepdim=True
        )

        cam = torch.sum(weights * activations, dim=0)

        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-7)

        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return cam.detach().cpu().squeeze().numpy()

    def forward_hook(self, module, input, output):
        self.forward_result = output.detach().squeeze(0)

    def backward_hook(self, module, grad_input, grad_output):
        self.backward_result = grad_output[0].detach().squeeze(0)

    def remove_hooks(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()

        if self.backward_handle is not None:
            self.backward_handle.remove()