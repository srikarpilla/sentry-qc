"""
app.py — Hugging Face Spaces entry point (ZeroGPU free tier)
Gradio 6 interface with @spaces.GPU wrapping TensorFlow inference.
"""
import json
import os
import time

import numpy as np
import spaces
import gradio as gr
import tensorflow as tf
from PIL import Image, ImageOps

# ── Suppress TF noise ──────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ── Load model + companion files at startup (CPU) ──────────────────────────────
MODEL_PATH      = "outputs/final_model.keras"
CLASS_NAMES_PATH = "outputs/class_names.json"
THRESHOLD_PATH  = "outputs/threshold.json"
IMG_SIZE        = (288, 288)
UNCERTAIN_BAND  = 0.10

print("Loading model …")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)          # ["defective", "good"]

with open(THRESHOLD_PATH) as f:
    thresh_data = json.load(f)
THRESHOLD = float(thresh_data.get("threshold", 0.58))

print(f"Model loaded — threshold={THRESHOLD}, classes={class_names}")

# Warm-up (keeps first real request fast)
_dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
model(_dummy, training=False)
print("Warm-up done.")


# ── Preprocessing helper ───────────────────────────────────────────────────────
def _preprocess(pil_img: Image.Image) -> np.ndarray:
    """Resize to 288×288, keep 0-255 float32 (EfficientNetB0 rescales internally)."""
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    tensor = tf.image.resize(tf.constant(arr), IMG_SIZE, method="bilinear")
    return np.expand_dims(tensor.numpy(), axis=0)   # (1, 288, 288, 3)


# ── Inference — must be wrapped with @spaces.GPU for ZeroGPU free tier ────────
@spaces.GPU
def _infer(batch: np.ndarray) -> np.ndarray:
    """Run one forward pass on GPU. Returns (N, 2) probability array."""
    return model(batch, training=False).numpy()


# ── Single-image prediction ────────────────────────────────────────────────────
def predict_single(image):
    """Gradio handler: PIL Image → formatted result string."""
    if image is None:
        return "⚠️ Please upload an image."

    pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    batch   = _preprocess(pil_img)

    t0    = time.perf_counter()
    probs = _infer(batch)[0]            # [p_defective, p_good]
    ms    = (time.perf_counter() - t0) * 1000

    p_defective = float(probs[0])
    p_good      = float(probs[1])

    low  = THRESHOLD - UNCERTAIN_BAND
    high = THRESHOLD + UNCERTAIN_BAND

    if low <= p_good <= high:
        verdict = "⚠️  NEEDS REVIEW (Uncertain)"
        note    = f"P(good) = {p_good:.1%} is near the decision boundary."
    elif p_good >= THRESHOLD:
        verdict = "✅  GOOD"
        note    = f"P(good) = {p_good:.1%} ≥ threshold {THRESHOLD:.0%}"
    else:
        verdict = "❌  DEFECTIVE"
        note    = f"P(good) = {p_good:.1%} < threshold {THRESHOLD:.0%}"

    return (
        f"### {verdict}\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| P(good) | **{p_good:.1%}** |\n"
        f"| P(defective) | **{p_defective:.1%}** |\n"
        f"| Threshold | {THRESHOLD} |\n"
        f"| Inference | {ms:.1f} ms |\n\n"
        f"_{note}_"
    )


# ── Batch prediction ───────────────────────────────────────────────────────────
def predict_batch(files):
    """Gradio handler: list of file paths → formatted markdown table."""
    if not files:
        return "⚠️ Please upload at least one image."

    rows = []
    for file_path in files[:16]:          # max 16 images
        name = os.path.basename(file_path)
        try:
            pil_img = Image.open(file_path)
            batch   = _preprocess(pil_img)
            probs   = _infer(batch)[0]
            p_good      = float(probs[1])
            p_defective = float(probs[0])
            low  = THRESHOLD - UNCERTAIN_BAND
            high = THRESHOLD + UNCERTAIN_BAND
            if low <= p_good <= high:
                label = "⚠️ UNCERTAIN"
            elif p_good >= THRESHOLD:
                label = "✅ GOOD"
            else:
                label = "❌ DEFECTIVE"
            rows.append(f"| {name} | {label} | {p_good:.1%} | {p_defective:.1%} |")
        except Exception as e:
            rows.append(f"| {name} | ⛔ ERROR | — | {e} |")

    header = (
        "| Filename | Label | P(good) | P(defective) |\n"
        "|----------|-------|---------|--------------|"
    )
    return header + "\n" + "\n".join(rows)


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Sentry QC — Automated Visual Inspection",
    theme=gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Sans"),
    ),
    css="""
    .gradio-container { max-width: 900px !important; margin: auto; }
    h1 { font-size: 2rem !important; font-weight: 700 !important; }
    """,
) as demo:

    gr.Markdown(
        """
        # 🛡️ Sentry QC — Automated Visual Inspection
        **EfficientNetB0** binary classifier trained on the MVTec-AD industrial anomaly dataset.
        Classifies product images as **Good** or **Defective** with 95% test accuracy.
        
        > **Note:** First inference may take ~15 s while the ZeroGPU warms up. Subsequent runs are fast.
        """
    )

    with gr.Tabs():
        # ── Tab 1: Single Image ────────────────────────────────────────────────
        with gr.Tab("🔍 Single Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label="Upload industrial product image",
                        type="numpy",
                        height=320,
                        sources=["upload", "clipboard"],
                    )
                    single_btn = gr.Button("Classify Image", variant="primary", size="lg")

                with gr.Column(scale=1):
                    single_output = gr.Markdown(
                        value="_Upload an image and click **Classify Image**._",
                        label="Result",
                    )

            single_btn.click(fn=predict_single, inputs=img_input, outputs=single_output)
            img_input.change(fn=predict_single, inputs=img_input, outputs=single_output)

        # ── Tab 2: Batch ───────────────────────────────────────────────────────
        with gr.Tab("📦 Batch (up to 16 images)"):
            batch_input = gr.File(
                label="Upload multiple images",
                file_count="multiple",
                file_types=["image"],
            )
            batch_btn    = gr.Button("Run Batch Inference", variant="primary", size="lg")
            batch_output = gr.Markdown(value="_Upload images and click **Run Batch Inference**._")

            batch_btn.click(fn=predict_batch, inputs=batch_input, outputs=batch_output)

    gr.Markdown(
        """
        ---
        **Model:** EfficientNetB0 · **Threshold:** 0.58 · **Uncertain band:** ±0.10  
        Trained on MVTec-AD (15 product categories). Not reliable on unrelated images.
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
