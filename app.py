"""
app.py – Hugging Face Spaces entry point.

Imports the Gradio demo from app/space_app.py and launches it.
Run locally:  python app.py
"""

from app.space_app import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
