"""
Visualization utilities.

Generates comparison plots (waveform, pitch contour, energy/RMS, spectrogram)
between reference and user recitation. Saved as PNGs next to the user recording.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive

import librosa
import librosa.display

from config import VIZ_CONFIG, AUDIO_CONFIG
from core.audio_processor import load_audio, extract_pitch_parselmouth, extract_rms_energy


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def plot_waveform_comparison(
    ref_path: Path,
    user_y: np.ndarray,
    user_sr: int,
    save_path: Path,
    title: str = "Waveform Comparison",
) -> Path:
    """Overlay reference and user waveforms (normalized time)."""
    ref_y, ref_sr = load_audio(ref_path, target_sr=user_sr)

    # Time axes
    t_ref = np.linspace(0, len(ref_y) / ref_sr, len(ref_y))
    t_user = np.linspace(0, len(user_y) / user_sr, len(user_y))

    fig, ax = plt.subplots(figsize=VIZ_CONFIG.waveform_figsize, dpi=VIZ_CONFIG.dpi)

    ax.plot(t_ref, ref_y, alpha=0.75, linewidth=0.8, label="Reference", color="#2E86AB")
    ax.plot(t_user, user_y, alpha=0.85, linewidth=0.9, label="Your recitation", color="#E94F37")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(0, max(t_ref.max(), t_user.max()) * 1.02)

    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=VIZ_CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_pitch_contour_comparison(
    ref_path: Path,
    user_y: np.ndarray,
    user_sr: int,
    save_path: Path,
    title: str = "Pitch Contour (Prosody)",
) -> Path:
    """Compare pitch trajectories — extremely valuable for Sanskrit tone work."""
    ref_y, ref_sr = load_audio(ref_path, target_sr=user_sr)

    ref_pitch = extract_pitch_parselmouth(ref_y, ref_sr)
    user_pitch = extract_pitch_parselmouth(user_y, user_sr)

    fig, ax = plt.subplots(figsize=VIZ_CONFIG.pitch_figsize, dpi=VIZ_CONFIG.dpi)

    # Reference
    ref_t = ref_pitch["time"]
    ref_p = ref_pitch["pitch"]
    ax.plot(ref_t, ref_p, alpha=0.7, linewidth=1.1, label="Reference", color="#2E86AB")

    # User
    user_t = user_pitch["time"]
    user_p = user_pitch["pitch"]
    ax.plot(user_t, user_p, alpha=0.85, linewidth=1.2, label="Your recitation", color="#E94F37")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(60, 420)  # reasonable chanting range

    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=VIZ_CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_energy_envelope_comparison(
    ref_path: Path,
    user_y: np.ndarray,
    user_sr: int,
    save_path: Path,
    title: str = "Energy Envelope (Volume & Phrasing)",
) -> Path:
    """RMS energy — shows phrasing, breath, volume consistency."""
    ref_y, ref_sr = load_audio(ref_path, target_sr=user_sr)

    ref_rms = extract_rms_energy(ref_y, hop_length=AUDIO_CONFIG.hop_length)
    user_rms = extract_rms_energy(user_y, hop_length=AUDIO_CONFIG.hop_length)

    t_ref = np.linspace(0, len(ref_y) / ref_sr, len(ref_rms))
    t_user = np.linspace(0, len(user_y) / user_sr, len(user_rms))

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=VIZ_CONFIG.dpi)

    ax.fill_between(t_ref, 0, ref_rms, alpha=0.35, color="#2E86AB", label="Reference")
    ax.fill_between(t_user, 0, user_rms, alpha=0.45, color="#E94F37", label="Your recitation")

    ax.plot(t_ref, ref_rms, linewidth=0.9, color="#2E86AB", alpha=0.9)
    ax.plot(t_user, user_rms, linewidth=0.95, color="#E94F37", alpha=0.95)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Energy")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=VIZ_CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_all_visualizations(
    ref_audio_path: Path,
    user_audio: np.ndarray,
    user_sr: int,
    base_output_path: Path,
    mantra_name: str = "Mantra",
) -> Dict[str, Path]:
    """
    Generate the full set of diagnostic plots for a practice session.
    Returns dict of {plot_type: saved_path}.
    """
    plots: Dict[str, Path] = {}

    # Waveform
    wf_path = base_output_path.with_name(base_output_path.stem + "_waveform.png")
    plots["waveform"] = plot_waveform_comparison(
        ref_audio_path, user_audio, user_sr, wf_path, f"{mantra_name} — Waveform"
    )

    # Pitch
    pitch_path = base_output_path.with_name(base_output_path.stem + "_pitch.png")
    plots["pitch"] = plot_pitch_contour_comparison(
        ref_audio_path, user_audio, user_sr, pitch_path, f"{mantra_name} — Pitch Contour"
    )

    # Energy
    energy_path = base_output_path.with_name(base_output_path.stem + "_energy.png")
    plots["energy"] = plot_energy_envelope_comparison(
        ref_audio_path, user_audio, user_sr, energy_path, f"{mantra_name} — Energy / Phrasing"
    )

    return plots
