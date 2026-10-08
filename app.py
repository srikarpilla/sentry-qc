import uvicorn
import gradio as gr
from app.main import app

# Mount a minimal Gradio interface at /gradio so Hugging Face registers the Gradio SDK,
# while your custom index.html frontend remains served at the root URL (/)
with gr.Blocks(title="Sentry QC") as demo:
    gr.Markdown("# Sentry QC API Backend Active")
    gr.Markdown("Your custom frontend is running at `/`.")

# Mount Gradio into your existing FastAPI application
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
