# src/gradcam.py
"""
Grad-CAM implementation for ResNet-style models.

Usage:
    from src.gradcam import GradCAM, make_heatmap, overlay_heatmap
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate_cam(image_tensor, class_idx)
    overlay = overlay_heatmap(pil_image, heatmap)
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # register hooks
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple; take the first element
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        return self.generate_cam(x, class_idx)

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        input_tensor: shape (1, C, H, W) on same device as model
        returns: heatmap numpy array (H, W) normalized 0..1
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # forward
        outputs = self.model(input_tensor)  # shape (1, num_classes)
        if class_idx is None:
            class_idx = int(torch.argmax(outputs, dim=1).item())

        # zero grads, backward on chosen class
        self.model.zero_grad()
        score = outputs[0, class_idx]
        score.backward(retain_graph=True)

        # gradients: (N, C, H, W), activations: (N, C, H, W)
        grads = self.gradients  # shape (1, C, H, W)
        acts = self.activations  # shape (1, C, H, W)

        # global average pooling on gradients -> weights (C)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # shape (1, C, 1, 1)
        # weighted sum of activations
        weighted = (weights * acts).sum(dim=1, keepdim=True)  # shape (1,1,H,W)
        cam = F.relu(weighted)
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        # normalize to 0..1
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))  # resize to input size
        return cam

# helper: transform PIL -> tensor normalized same as training
def pil_to_tensor(pil_img: Image.Image, image_size: int = 224, device: torch.device = None) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    t = tf(pil_img).unsqueeze(0)
    if device:
        t = t.to(device)
    return t

# helper: make colour heatmap from cam (0..1)
def make_heatmap(cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    # cam expected HxW, values 0..1
    heat = np.uint8(255 * cam)
    heat = cv2.applyColorMap(heat, colormap)  # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = heat.astype(np.float32) / 255.0
    return heat  # HxWx3 float32

def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    # cam: HxW normalized 0..1
    img = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    heat = make_heatmap(cam)
    overlay = (1 - alpha) * img + alpha * heat
    overlay = np.clip(overlay, 0, 1)
    overlay = np.uint8(overlay * 255)
    return Image.fromarray(overlay)

# convenience: get target layer for ResNet-18 (last conv)
def resnet_last_conv(model: torch.nn.Module):
    # For torchvision.models.resnet, last conv is model.layer4[-1].conv2
    try:
        return model.layer4[-1].conv2
    except Exception:
        # fallback: search for last nn.Conv2d
        last = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last = m
        return last
