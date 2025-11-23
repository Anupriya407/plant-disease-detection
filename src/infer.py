# src/infer.py
"""
Simple inference utility for Plant Disease Detection.

Usage (from project root):
    python src\infer.py --image "data/samples/my_leaf.jpg"

Or import functions:
    from src.infer import load_model, predict_image
"""

from pathlib import Path
import torch
from torchvision import models, transforms
from PIL import Image
import json
import argparse

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
IMAGE_SIZE = 224

# --- transforms ---
def get_transform(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

# --- load model ---
def load_model(model_path: str = None, device: torch.device = None):
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt = torch.load(model_path, map_location=device)

    # Expect checkpoint to contain "model_state" and "classes"
    classes = ckpt.get("classes")
    if classes is None:
        raise KeyError("Saved checkpoint does not contain 'classes' key.")

    # create model and load state
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, classes, device

# --- predict single image ---
def predict_image(image_path: str, model, classes, device=None, topk=3):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    transform = get_transform()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu()
    topk_probs, topk_idx = torch.topk(probs, k=min(topk, probs.shape[0]))
    topk_probs = topk_probs.tolist()
    topk_idx = topk_idx.tolist()
    results = [{"label": classes[i], "confidence": float(topk_probs[idx])} for idx, i in enumerate(topk_idx)]
    return results

# --- CLI support ---
def main():
    parser = argparse.ArgumentParser(description="Run inference on a single leaf image")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default=None, help="Optional model path (overrides default)")
    parser.add_argument("--topk", type=int, default=3, help="Return top-k predictions")
    args = parser.parse_args()

    model, classes, device = load_model(args.model)
    print(f"Loaded model on device: {device}. Number of classes: {len(classes)}")

    res = predict_image(args.image, model, classes, device=device, topk=args.topk)
    print("Top predictions:")
    for r in res:
        print(f"  {r['label']}  —  {r['confidence']:.4f}")

if __name__ == "__main__":
    main()
