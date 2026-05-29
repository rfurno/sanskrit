"""Core analysis modules: audio, phoneme alignment, evaluation, feedback.

Heavy dependencies (librosa, parselmouth, pyaudio, etc.) are imported only
when you explicitly import the individual submodules.
"""
# Do NOT do star-imports here — they would pull heavy scientific/audio libs
# at package import time. Users should do:
#   from core.evaluator import evaluate_recitation
# or import the top-level main CLI.

