import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, use_deep_conv):
        self.model = model.eval()
        self.activations = []
        self.gradients = []

        # Hook into last conv layer
        if use_deep_conv:
            layer = self.model.conv3[0]
        else:
            layer = self.model.convs[6]
        layer.register_forward_hook(lambda m, inp, out: self.activations.append(out))
        layer.register_full_backward_hook(
            lambda m, grad_in, grad_out: self.gradients.append(grad_out[0])
        )

    def __call__(self, input_dict, args):
        # Forward
        output = self.model(input_dict)  # [1, A, atoms]
        support = torch.linspace(
            args.V_min, args.V_max, args.atoms, device=output.device
        )
        q_dist = output[0]  # [action, atoms]
        q_vals = (q_dist * support).sum(dim=1)  # [action]
        idx = q_vals.argmax().item()

        # Backward
        self.model.zero_grad()
        q_vals[idx].backward(retain_graph=True)

        # Grad-CAM map
        grad = self.gradients.pop().detach()  # [1, C, H, W]
        act = self.activations.pop().detach()  # [1, C, H, W]
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, C,1,1]
        cam = (weights * act).sum(dim=1)[0]  # [H, W]
        cam = F.relu(cam)
        cam -= cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam /= maximum
        return cam.cpu().numpy(), idx
