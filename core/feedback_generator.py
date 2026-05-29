"""
Rich feedback generator for SanskritMantraPronunciationCoach.

Produces beautiful, encouraging, and pedagogically useful terminal output
using the `rich` library. Tailored to highlight Sanskrit phonetic challenges.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from config import THRESHOLDS, FEEDBACK_CONFIG
from core.evaluator import EvaluationResult, AttributeScore


console = Console()


def _color_for_score(score: float) -> str:
    if score >= THRESHOLDS.excellent:
        return "bright_green"
    elif score >= THRESHOLDS.good:
        return "green"
    elif score >= THRESHOLDS.fair:
        return "yellow"
    elif score >= THRESHOLDS.poor:
        return "orange_red1"
    return "red"


def _render_score_badge(score: float) -> Text:
    label = THRESHOLDS.get_label(score)
    color = _color_for_score(score)
    t = Text(f" {score:.0f} ", style=f"bold {color} on black")
    t.append(f" {label}", style=color)
    return t


def print_header(result: EvaluationResult) -> None:
    """Print mantra header + overall score."""
    console.print()
    title = Text("🕉️  ", style="yellow")
    title.append(result.mantra_name, style="bold white")
    title.append(f"  —  {result.timestamp}", style="dim")

    console.print(Panel(title, border_style="magenta", box=box.HEAVY))

    overall_text = Text("Overall Score: ", style="bold")
    overall_text.append(f"{result.overall_score:.1f}/100", style=f"bold {_color_for_score(result.overall_score)}")
    overall_text.append(f"   ({THRESHOLDS.get_label(result.overall_score)})", style=_color_for_score(result.overall_score))

    console.print(overall_text)
    console.print(f"[dim]{result.summary}[/dim]")
    console.print()


def print_score_table(result: EvaluationResult) -> None:
    """Main per-attribute scorecard."""
    table = Table(
        title="📊 Detailed Evaluation",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("Attribute", style="cyan", no_wrap=True)
    table.add_column("Score", justify="center", width=14)
    table.add_column("Weighted", justify="right", style="dim")
    table.add_column("Feedback", style="white", min_width=42)

    for key, attr in result.attribute_scores.items():
        score_text = _render_score_badge(attr.score)
        weight_str = f"{attr.weighted_contribution:.1f}"

        table.add_row(
            attr.name,
            score_text,
            weight_str,
            attr.comment[:110] + ("…" if len(attr.comment) > 110 else ""),
        )

    console.print(table)
    console.print()


def print_strengths_weaknesses(result: EvaluationResult) -> None:
    """Highlight what went well and what needs work."""
    if result.strengths:
        s = Text("✓ Strengths:  ", style="bold green")
        s.append("  •  ".join(result.strengths), style="green")
        console.print(s)

    if result.weaknesses:
        w = Text("✗ Focus Areas: ", style="bold orange_red1")
        w.append("  •  ".join(result.weaknesses), style="orange_red1")
        console.print(w)

    console.print()


def print_sanskrit_tips(result: EvaluationResult) -> None:
    """Sanskrit-specific guidance."""
    if not result.sanskrit_tips or not FEEDBACK_CONFIG.show_phonetic_tips:
        return

    tips_panel = Panel(
        "\n".join(f"• {tip}" for tip in result.sanskrit_tips[: FEEDBACK_CONFIG.max_tips_per_session]),
        title="[bold yellow]Sanskrit Phonetic Guidance[/bold yellow]",
        border_style="yellow",
        box=box.SQUARE,
    )
    console.print(tips_panel)
    console.print()


def print_alignment_info(result: EvaluationResult) -> None:
    """Show technical alignment summary (for advanced users)."""
    if not result.alignment:
        return

    a = result.alignment
    tech = (
        f"DTW distance: [bold]{a['dtw_distance']}[/bold]   |   "
        f"Reference: {a['ref_duration']:.1f}s   →   Your take: {a['user_duration']:.1f}s"
    )
    console.print(Panel(tech, title="Alignment Details", border_style="dim", box=box.MINIMAL))
    console.print()


def print_final_encouragement(result: EvaluationResult) -> None:
    """End on an encouraging, culturally respectful note."""
    if result.overall_score >= 85:
        msg = "Excellent work. Your pronunciation honors the mantra's vibrational quality. Keep refining the subtle mātrā."
    elif result.overall_score >= 72:
        msg = "Very good progress. The foundation is solid — now polish vowel lengths and breath placement."
    elif result.overall_score >= 58:
        msg = "Solid attempt. Focus on one or two dimensions per session (e.g., long ī and visarga). Repetition builds clarity."
    else:
        msg = "Mantras reward patient, mindful practice. Listen to the reference many times, then record short phrases first."

    console.print(Panel(f"[italic]{msg}[/italic]", border_style="green", box=box.ROUNDED))
    console.print()


def print_full_feedback(result: EvaluationResult, show_technical: bool = False) -> None:
    """Complete beautiful feedback rendering."""
    print_header(result)
    print_score_table(result)
    print_strengths_weaknesses(result)
    print_sanskrit_tips(result)
    if show_technical:
        print_alignment_info(result)
    print_final_encouragement(result)


def print_mantra_card(mantra_data: dict) -> None:
    """Pretty display of a mantra when user selects --practice or --list."""
    console.print()
    header = Text(mantra_data.get("name", "Mantra"), style="bold yellow")
    header.append(f"  ({mantra_data.get('id', '')})", style="dim")

    console.print(Panel(header, border_style="yellow"))

    deva = Text(mantra_data.get("devanagari", ""), style="bold white on black", justify="center")
    console.print(deva)
    console.print()

    iast = Text("IAST: ", style="dim")
    iast.append(mantra_data.get("iast", ""), style="italic cyan")
    console.print(iast)

    if mantra_data.get("english_translation"):
        trans = Text("Meaning: ", style="dim")
        trans.append(mantra_data.get("english_translation"), style="green")
        console.print(trans)

    console.print()
