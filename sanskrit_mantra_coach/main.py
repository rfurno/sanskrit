#!/usr/bin/env python3
"""
SanskritMantraPronunciationCoach

A sophisticated tool for learning authentic Sanskrit mantra pronunciation.

Usage examples:
    python -m sanskrit_mantra_coach.main --help
    python -m sanskrit_mantra_coach.main list-mantras
    python -m sanskrit_mantra_coach.main practice gayatri_mantra
    python -m sanskrit_mantra_coach.main evaluate gayatri_mantra --audio path/to/recording.wav
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich import print as rprint

# Local imports
from config import (
    MANTRAS_DIR,
    REFERENCES_DIR,
    USER_RECORDINGS_DIR,
    REPORTS_DIR,
    ASR_CONFIG,
)
from core.audio_processor import (
    AudioRecorder,
    play_audio,
    load_audio,
    save_audio,
    ensure_reference_audio,
)
from core.evaluator import evaluate_recitation, EvaluationResult
from core.feedback_generator import (
    print_full_feedback,
    print_mantra_card,
    console,
)
from core.phoneme_aligner import load_mantra, list_available_mantras
from utils.helpers import save_user_recording, get_latest_user_recording
from utils.visualizations import generate_all_visualizations

# Optional ASR
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

app = typer.Typer(
    name="sanskrit-coach",
    help="🕉️  Sanskrit Mantra Pronunciation Coach — precise, compassionate feedback for authentic chanting.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# =============================================================================
# HELPERS
# =============================================================================

def _get_asr_model():
    """Lazy-load Whisper model only when needed."""
    if not ASR_CONFIG.enabled:
        return None
    if not FASTER_WHISPER_AVAILABLE:
        rprint("[yellow]⚠️  faster-whisper not installed. ASR disabled.[/yellow]")
        return None
    try:
        model = WhisperModel(
            ASR_CONFIG.model_size,
            device=ASR_CONFIG.device,
            compute_type=ASR_CONFIG.compute_type,
        )
        return model
    except Exception as e:
        rprint(f"[red]Failed to load Whisper model: {e}[/red]")
        return None


def _transcribe(y: "np.ndarray", sr: int, model) -> Optional[str]:
    if model is None:
        return None
    # Resample to 16k if needed (Whisper requirement)
    import librosa
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    segments, info = model.transcribe(y, language=ASR_CONFIG.language, beam_size=ASR_CONFIG.beam_size)
    text = " ".join(seg.text for seg in segments).strip()
    return text or None


def _maybe_play_reference(mantra_id: str) -> None:
    ref_path = ensure_reference_audio(mantra_id)
    if Confirm.ask("🔊 Play reference recording now?", default=True):
        y, sr = load_audio(ref_path)
        play_audio(y, sr)


# =============================================================================
# COMMANDS
# =============================================================================

@app.command("list-mantras")
def list_mantras():
    """List all available mantras in the database."""
    mantras = list_available_mantras()
    if not mantras:
        rprint("[red]No mantras found in mantras/ directory.[/red]")
        raise typer.Exit(1)

    rprint("\n[bold magenta]📿 Available Mantras[/bold magenta]\n")
    for mid in mantras:
        try:
            m = load_mantra(mid)
            rprint(f"  [cyan]{mid}[/cyan]  —  [white]{m.name}[/white]  ({m.approx_duration:.0f}s)")
            rprint(f"      [dim]{m.devanagari[:65]}...[/dim]")
        except Exception as e:
            rprint(f"  [red]{mid}[/red]  (error loading: {e})")
    rprint()


@app.command("practice")
def practice(
    mantra_id: str = typer.Argument(..., help="Mantra ID (e.g. gayatri_mantra)"),
    record_seconds: int = typer.Option(50, "--seconds", "-s", help="Max recording length"),
    play_ref: bool = typer.Option(True, "--play-ref/--no-play-ref", help="Play reference before recording"),
    save_visuals: bool = typer.Option(True, "--visuals/--no-visuals", help="Generate comparison plots"),
    show_technical: bool = typer.Option(False, "--technical", help="Show DTW/alignment numbers"),
):
    """
    Full practice loop: show mantra → play reference → record your voice → evaluate → feedback + visuals.
    """
    try:
        mantra = load_mantra(mantra_id)
    except FileNotFoundError:
        rprint(f"[red]Mantra '{mantra_id}' not found.[/red]")
        rprint(f"Available: {', '.join(list_available_mantras())}")
        raise typer.Exit(1)

    print_mantra_card(mantra.raw_json)

    ref_path = ensure_reference_audio(mantra_id)

    if play_ref:
        _maybe_play_reference(mantra_id)

    # Record
    rprint("\n[bold]When you are ready, press Enter to start recording...[/bold]")
    input()
    recorder = AudioRecorder()
    try:
        user_y, user_sr = recorder.record(max_seconds=record_seconds)
    finally:
        recorder.close()

    if len(user_y) < user_sr * 3:
        rprint("[red]Recording too short. Please try again.[/red]")
        raise typer.Exit(1)

    # Save user attempt
    user_path = save_user_recording(user_y, user_sr, mantra_id)
    rprint(f"\n💾 Saved your attempt to [cyan]{user_path}[/cyan]")

    # Optional ASR (if enabled in config)
    asr_text = None
    model = _get_asr_model()
    if model:
        rprint("[dim]Transcribing with Whisper (this may take a moment on CPU)...[/dim]")
        asr_text = _transcribe(user_y, user_sr, model)
        if asr_text:
            rprint(f"[yellow]Heard (ASR):[/yellow] {asr_text[:180]}")

    # Evaluate
    rprint("\n[bold magenta]🔬 Analyzing pronunciation...[/bold magenta]")
    result: EvaluationResult = evaluate_recitation(
        mantra_id=mantra_id,
        user_audio=user_y,
        user_sr=user_sr,
        user_audio_path=user_path,
        asr_transcript=asr_text,
        play_reference=False,
    )

    # Rich feedback
    print_full_feedback(result, show_technical=show_technical)

    # Visualizations
    if save_visuals:
        rprint("\n[bold]📈 Generating diagnostic visualizations...[/bold]")
        plots = generate_all_visualizations(
            ref_path, user_y, user_sr, user_path, mantra.name
        )
        for name, p in plots.items():
            rprint(f"   • {name}: [cyan]{p}[/cyan]")

    # Report
    report_path = result.save_report()
    rprint(f"\n📄 Detailed JSON report saved to [cyan]{report_path}[/cyan]\n")

    if Confirm.ask("Practice the same mantra again?", default=False):
        practice(mantra_id, record_seconds, play_ref, save_visuals, show_technical)


@app.command("evaluate")
def evaluate(
    mantra_id: str = typer.Argument(..., help="Mantra ID"),
    audio: Optional[Path] = typer.Option(
        None, "--audio", "-a", help="Path to pre-recorded user WAV file (instead of live recording)"
    ),
    play_ref: bool = typer.Option(False, "--play-ref", help="Play reference audio before analysis"),
    technical: bool = typer.Option(False, "--technical", help="Include alignment metrics"),
):
    """
    Evaluate an existing recording against the reference (no new recording).
    """
    try:
        mantra = load_mantra(mantra_id)
    except FileNotFoundError:
        rprint(f"[red]Mantra not found: {mantra_id}[/red]")
        raise typer.Exit(1)

    if audio is None:
        # Try latest user recording
        latest = get_latest_user_recording(mantra_id)
        if latest is None:
            rprint("[red]No --audio provided and no previous recordings found for this mantra.[/red]")
            rprint("Use `practice` command to record, or provide --audio path/to/file.wav")
            raise typer.Exit(1)
        audio = latest
        rprint(f"Using most recent recording: [cyan]{audio}[/cyan]")

    if not audio.exists():
        rprint(f"[red]Audio file not found: {audio}[/red]")
        raise typer.Exit(1)

    user_y, user_sr = load_audio(audio)

    if play_ref:
        _maybe_play_reference(mantra_id)

    rprint(f"\n[bold]Evaluating [cyan]{audio.name}[/cyan] against {mantra.name} reference...[/bold]\n")

    result = evaluate_recitation(
        mantra_id=mantra_id,
        user_audio=user_y,
        user_sr=user_sr,
        user_audio_path=audio,
        asr_transcript=None,
    )
    print_full_feedback(result, show_technical=technical)

    report_path = result.save_report()
    rprint(f"\n📄 Report: [cyan]{report_path}[/cyan]")


@app.command("record-reference")
def record_reference(
    mantra_id: str = typer.Argument(..., help="Mantra ID to create reference for"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Custom output path for the WAV"),
):
    """
    Record a high-quality reference version of a mantra (for teachers or advanced users).
    This becomes the new gold-standard for evaluation.
    """
    try:
        mantra = load_mantra(mantra_id)
    except FileNotFoundError:
        rprint(f"[red]Unknown mantra: {mantra_id}[/red]")
        raise typer.Exit(1)

    rprint(f"\n[bold yellow]Recording NEW REFERENCE for {mantra.name}[/bold yellow]")
    rprint("[dim]Please chant at your clearest, most traditional pace. This will be used for all future evaluations.[/dim]\n")

    if not Confirm.ask("Ready to record the reference?", default=True):
        raise typer.Exit(0)

    recorder = AudioRecorder()
    try:
        y, sr = recorder.record(max_seconds=65)
    finally:
        recorder.close()

    if output is None:
        output = REFERENCES_DIR / f"{mantra_id}.wav"
    save_audio(output, y, sr)
    rprint(f"\n✅ New reference saved to [green]{output}[/green]")
    rprint("[yellow]All future evaluations for this mantra will use this file.[/yellow]\n")


@app.command("info")
def info(mantra_id: Optional[str] = typer.Argument(None)):
    """Show detailed information about a mantra (or all if none given)."""
    if mantra_id is None:
        list_mantras()
        return
    try:
        m = load_mantra(mantra_id)
    except FileNotFoundError:
        rprint(f"[red]Not found: {mantra_id}[/red]")
        raise typer.Exit(1)

    print_mantra_card(m.raw_json)
    if m.guidance:
        rprint("[bold yellow]Sanskrit Guidance:[/bold yellow]")
        for g in m.guidance:
            rprint(f"  • {g}")
    if m.common_mistakes:
        rprint("\n[bold orange_red1]Common Mistakes:[/bold orange_red1]")
        for c in m.common_mistakes:
            rprint(f"  • {c}")
    rprint()


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    app()
