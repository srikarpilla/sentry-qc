"""
app.py — Hugging Face Spaces entry point
Mounts the Sentry QC FastAPI app inside a minimal Gradio 6 shell.
The full custom frontend is served at the root URL (/).
"""
import gradio as gr
import uvicorn

from app.main import app as fastapi_app

# Minimal Gradio interface — required so HF Spaces detects a valid Gradio app
with gr.Blocks(title="Sentry QC") as demo:
    gr.Markdown(
        """
        # 🛡️ Sentry QC — Automated Visual Inspection
        The full web application is running at the **root URL [`/`](/)**.

        You can also explore the API at [`/docs`](/docs).
        """
    )

# Mount Gradio at /gradio; FastAPI handles /, /predict, /health, etc.
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
