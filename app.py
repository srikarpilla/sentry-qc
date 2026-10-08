"""
app.py — Hugging Face Spaces entry point (CPU basic)
Mounts the Sentry QC FastAPI app inside a minimal Gradio 6 shell.
The full custom frontend is served at the root URL (/).
"""
import gradio as gr
import uvicorn

from app.main import app as fastapi_app

# Minimal Gradio interface mounted at /gradio
# FastAPI handles everything else: /, /predict, /health, /docs
with gr.Blocks(title="Sentry QC") as demo:
    gr.Markdown(
        """
        # 🛡️ Sentry QC — Automated Visual Inspection
        The full web application is running at **[/](/)**.
        API docs are available at **[/docs](/docs)**.
        """
    )

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
