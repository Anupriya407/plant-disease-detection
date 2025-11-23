# app/streamlit_app.py
"""
Streamlit demo with Grad-CAM explainability (Original | Heatmap | Overlay)
Run from project root (with venv active):
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import streamlit as st
from PIL import Image
import torch

from src.infer import load_model
from src.gradcam import GradCAM, pil_to_tensor, overlay_heatmap, make_heatmap, resnet_last_conv

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best_model.pth"
DEFAULT_SAMPLE = str(Path(__file__).resolve().parent / "sample.jpg")

@st.cache_resource
def get_model_and_classes(model_path: str = None):
    model, classes, device = load_model(model_path or str(DEFAULT_MODEL))
    return model, classes, device

def run_inference_and_gradcam(pil_img: Image.Image, model, classes, device, topk=3):
    # inference
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    model.eval()
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outs = model(tensor)
        probs = torch.softmax(outs, dim=1)[0].cpu()
    topk_probs, topk_idx = torch.topk(probs, k=min(topk, probs.shape[0]))

    results = [{"label": classes[i], "confidence": float(topk_probs[idx])} for idx, i in enumerate(topk_idx.tolist())]

    # Grad-CAM for top1
    target_layer = resnet_last_conv(model)
    cam = GradCAM(model, target_layer)
    # generate cam: pass tensor same device
    cam_mask = cam.generate_cam(tensor, class_idx=topk_idx[0].item())  # HxW normalized
    heatmap_np = make_heatmap(cam_mask)
    overlay_img = overlay_heatmap(pil_img.resize((224,224)), cam_mask, alpha=0.5)

    return results, heatmap_np, overlay_img

def main():
    st.set_page_config(page_title="Plant Disease Detector (Grad-CAM)", layout="wide")
    st.title("🌿 Plant Disease Detection — Demo + Grad-CAM")

    # Load model
    with st.spinner("Loading model..."):
        try:
            model, classes, device = get_model_and_classes(str(DEFAULT_MODEL))
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    st.sidebar.header("Options")
    topk = st.sidebar.slider("Top-k predictions", min_value=1, max_value=5, value=3)
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM explainability", value=True)
    uploaded = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])

    use_sample = st.button("Use sample image")
    if use_sample:
        try:
            sample_img = Image.open(DEFAULT_SAMPLE).convert("RGB")
            uploaded_img = sample_img
        except Exception as e:
            st.error(f"Could not load sample image at {DEFAULT_SAMPLE}: {e}")
            uploaded_img = None
    else:
        uploaded_img = None

    if uploaded is not None:
        try:
            uploaded_img = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error(f"Failed to read uploaded image: {e}")
            uploaded_img = None

    if uploaded_img is None:
        st.info("Upload an image or click 'Use sample image' to try a demo.")
        return

    # Run inference (and gradcam if needed)
    with st.spinner("Running inference..."):
        try:
            results, heatmap_np, overlay_img = run_inference_and_gradcam(uploaded_img, model, classes, device, topk=topk)
        except Exception as e:
            st.error(f"Inference / Grad-CAM failed: {e}")
            return

    # Layout: three columns for original, heatmap, overlay if Grad-CAM enabled
    if show_gradcam:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Original")
            st.image(uploaded_img, use_column_width=True)
        with c2:
            st.subheader("Grad-CAM (heatmap)")
            # heatmap_np is float HxWx3 0..1, convert to uint8
            import numpy as np
            heat_uint8 = (np.clip(heatmap_np, 0, 1) * 255).astype("uint8")
            st.image(heat_uint8, use_column_width=True)
        with c3:
            st.subheader("Overlay")
            st.image(overlay_img, use_column_width=True)
    else:
        st.subheader("Input")
        st.image(uploaded_img, use_column_width=True)

    # Predictions panel
    st.markdown("---")
    st.subheader("Predictions")
    for r in results:
        st.write(f"- **{r['label']}**  —  {r['confidence']:.4f}")

    st.caption("Grad-CAM highlights the regions the model used to make its decision. Use this to explain predictions during your demo.")

if __name__ == "__main__":
    main()
