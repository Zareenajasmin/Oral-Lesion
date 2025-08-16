import io
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ------------------------------
# App config
# ------------------------------
st.set_page_config(page_title="Oral Lesion Segmentation", layout="wide")
st.title("🦷 Oral Lesion Segmentation (SegFormer-B2)")
st.write("Upload an intraoral image to segment suspected lesion regions. Model: SegFormer-B2 fine-tuned (2 classes).")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    overlay_alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
    st.caption("Adjust threshold/opacity to tune the visualization.")

# ------------------------------
# Load model + processor (CPU/GPU auto)
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_processor(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    # Load your fine-tuned weights
    state = torch.load(checkpoint_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        # If the file saved the whole model (not state_dict)
        model = state
    model.eval().to(device)
    return model, processor, device

# Path to your uploaded checkpoint (place best_model.pth in same folder as app.py)
MODEL_PATH = "best_model.pth"

try:
    model, processor, device = load_model_and_processor(MODEL_PATH)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Could not load model from '{MODEL_PATH}'. Error: {e}")
    st.stop()

# ------------------------------
# Utilities
# ------------------------------

def pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def predict_mask(image_pil: Image.Image, conf: float = 0.5) -> np.ndarray:
    """Run the model and return a binary mask (H,W) with 1=lesion, 0=background."""
    orig_w, orig_h = image_pil.size
    image_np = pil_to_numpy_rgb(image_pil)

    inputs = processor(images=image_np, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # [B, num_labels, h, w]
        # Upsample to original size
        upsampled = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        probs = upsampled.softmax(dim=1)[0]  # [2, H, W]
        lesion_prob = probs[1]               # class 1 = lesion
        mask = (lesion_prob >= conf).to(torch.uint8).cpu().numpy()  # (H,W)
    return mask


def overlay_mask(image_pil: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay red mask on the original image with given alpha."""
    base = image_pil.convert("RGBA")
    h, w = mask.shape
    # Create red overlay where mask==1
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., 0] = 255  # R
    overlay[..., 1] = 0    # G
    overlay[..., 2] = 0    # B
    overlay[..., 3] = (mask * int(alpha * 255)).astype(np.uint8)  # A
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    return Image.alpha_composite(base, overlay_img).convert("RGB")

# ------------------------------
# UI: file uploader
# ------------------------------
uploaded = st.file_uploader("Upload an intraoral image (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = Image.open(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with st.spinner("Running segmentation …"):
        mask = predict_mask(image, conf=conf_thresh)
        overlay = overlay_mask(image, mask, alpha=overlay_alpha)

    with col2:
        st.subheader("Segmentation Overlay")
        st.image(overlay, use_container_width=True)

    st.caption("Class 1 (red) = lesion; Class 0 = background.")
else:
    st.info("Upload an image to get started.")
