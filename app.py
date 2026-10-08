import uvicorn

# Hugging Face Spaces requires at least one @spaces.GPU decorated function
# even on CPU-only spaces. This satisfies the startup check.
try:
    import spaces

    @spaces.GPU(duration=0)
    def _hf_gpu_placeholder():
        pass

except Exception:
    # Running locally without the spaces package — safe to ignore
    pass

from app.main import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
