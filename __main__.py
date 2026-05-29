"""
Entry point for `python -m sanskrit_mantra_coach`

This allows running the tool directly as a module from the repository root:
    python -m sanskrit_mantra_coach

Delegates to the Typer CLI in main.py.
"""

from .main import app

if __name__ == "__main__":
    app()
