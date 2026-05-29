"""
SanskritMantraPronunciationCoach - Configuration

All evaluation weights, thresholds, audio parameters, and Sanskrit phonetic
definitions are centralized here for easy tuning and experimentation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
MANTRAS_DIR = PROJECT_ROOT / "mantras"
AUDIO_DIR = PROJECT_ROOT / "audio"
REFERENCES_DIR = AUDIO_DIR / "references"
USER_RECORDINGS_DIR = AUDIO_DIR / "user_recordings"
REPORTS_DIR = PROJECT_ROOT / "reports"  # For saved visualizations + feedback

for d in [REFERENCES_DIR, USER_RECORDINGS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# AUDIO PARAMETERS
# =============================================================================
@dataclass
class AudioConfig:
    # Target sample rate for analysis (Whisper uses 16000 internally)
    sample_rate: int = 22050
    # For Whisper transcription (if enabled)
    whisper_sample_rate: int = 16000

    # Feature extraction
    n_mfcc: int = 20
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048

    # Recording
    channels: int = 1
    chunk_size: int = 1024
    format: int = 8  # pyaudio.paInt16 (value 8)
    record_default_seconds: int = 45  # Max for long mantras; user can stop early

    # Preprocessing
    trim_silence_top_db: int = 25
    normalize: bool = True

    # Playback
    playback_chunk_size: int = 2048


AUDIO_CONFIG = AudioConfig()


# =============================================================================
# EVALUATION WEIGHTS (must sum to ~1.0; normalized at runtime)
# =============================================================================
# These are the primary levers for tuning feedback emphasis.
# Sanskrit mantra chanting prioritizes: vowel length, rhythm, stability, breath.
EVALUATION_WEIGHTS: Dict[str, float] = {
    "pronunciation_accuracy": 0.22,   # Phoneme / acoustic fidelity (Sanskrit sounds)
    "vowel_length_accuracy": 0.18,    # Critical in Sanskrit (mātrā)
    "tone_pitch_stability": 0.15,     # Steady tone, minimal jitter in sustained vowels
    "speed_tempo": 0.12,              # Overall duration + rhythm alignment
    "smoothness_fluency": 0.10,       # Pauses, hesitations, flow between phrases
    "volume_consistency": 0.08,       # Steady amplitude (no sudden drops)
    "breath_control": 0.08,           # Appropriate breath placement & length
    "overall_similarity": 0.07,       # Global acoustic + prosodic similarity (DTW)
}

# Ensure weights sum to 1.0
_total_w = sum(EVALUATION_WEIGHTS.values())
EVALUATION_WEIGHTS = {k: v / _total_w for k, v in EVALUATION_WEIGHTS.items()}


# =============================================================================
# SCORING THRESHOLDS (used for qualitative labels and color)
# =============================================================================
@dataclass
class ScoreThresholds:
    excellent: float = 90.0
    good: float = 75.0
    fair: float = 60.0
    poor: float = 45.0
    # Below poor is "needs work"

    def get_label(self, score: float) -> str:
        if score >= self.excellent:
            return "Excellent"
        elif score >= self.good:
            return "Good"
        elif score >= self.fair:
            return "Fair"
        elif score >= self.poor:
            return "Needs Improvement"
        return "Significant Work Needed"


THRESHOLDS = ScoreThresholds()


# =============================================================================
# METRIC-SPECIFIC THRESHOLDS & TUNING
# =============================================================================
@dataclass
class MetricConfig:
    # Vowel length (mātrā) - very important for Sanskrit
    vowel_length_tolerance_ratio: float = 0.35   # Allowed deviation from expected
    long_vowel_min_duration_ms: int = 280
    short_vowel_max_duration_ms: int = 220

    # Pitch / Tone stability
    pitch_stability_max_jitter_cents: float = 45.0   # Good chanting stays steady
    pitch_octave_jump_penalty: float = 25.0

    # Tempo
    tempo_tolerance_percent: float = 18.0   # % deviation from reference total duration
    rhythm_segment_tolerance: float = 0.30

    # Fluency / Smoothness
    max_acceptable_pause_ms: int = 650
    unnatural_pause_penalty_per: float = 8.0
    hesitation_energy_drop_db: float = 18.0

    # Volume
    volume_variation_max_db: float = 9.0   # std of short-term RMS

    # Breath control
    breath_min_silence_ms: int = 220
    breath_max_natural_ms: int = 1100
    breath_in_word_penalty: float = 15.0
    max_breaths_per_phrase: int = 1

    # DTW / Acoustic similarity (lower distance = better)
    dtw_good_threshold: float = 0.65
    dtw_excellent_threshold: float = 0.42

    # ASR / Text (if enabled). Lower WER better.
    asr_weight_in_pronunciation: float = 0.25  # How much text similarity affects phoneme score


METRIC_CONFIG = MetricConfig()


# =============================================================================
# SANSKRIT PHONETIC INVENTORY (for analysis and feedback)
# =============================================================================
# Focused on distinctions critical for correct mantra pronunciation.

SANSKRIT_VOWELS: Dict[str, Dict] = {
    "short": {"a", "i", "u", "ṛ", "ḷ"},
    "long": {"ā", "ī", "ū", "ṝ", "e", "ai", "o", "au"},
    "diphthongs": {"ai", "au", "e", "o"},
    "nasalized": {"aṃ", "iṃ", "uṃ", "āṃ", "īṃ", "ūṃ", "eṃ", "oṃ"},
}

SANSKRIT_CONSONANTS: Dict[str, List[str]] = {
    "velar": ["k", "kh", "g", "gh", "ṅ"],
    "palatal": ["c", "ch", "j", "jh", "ñ"],
    "retroflex": ["ṭ", "ṭh", "ḍ", "ḍh", "ṇ"],
    "dental": ["t", "th", "d", "dh", "n"],
    "labial": ["p", "ph", "b", "bh", "m"],
    "semivowels": ["y", "r", "l", "v"],
    "sibilants": ["ś", "ṣ", "s"],
    "aspirates": ["kh", "gh", "ch", "jh", "ṭh", "ḍh", "th", "dh", "ph", "bh"],
    "unaspirated_voiced": ["g", "j", "ḍ", "d", "b"],
    "unaspirated_voiceless": ["k", "c", "ṭ", "t", "p"],
    "nasals": ["ṅ", "ñ", "ṇ", "n", "m"],
}

# Critical pronunciation pitfalls for learners (used in feedback)
SANSKRIT_PRONUNCIATION_NOTES: Dict[str, str] = {
    "aspiration": "Aspirated consonants (kh, gh, th, dh, ph, bh, etc.) require a clear puff of air. Do not pronounce like English 'k' or 't'.",
    "retroflex": "Retroflex sounds (ṭ, ḍ, ṇ, ṣ, ṛ) are pronounced with the tongue tip curled back toward the hard palate. Not the same as dental t/d.",
    "vowel_length": "Vowel length (mātrā) is phonemic. ā is held roughly twice as long as a. This is one of the most common errors.",
    "visarga": "Visarga (ḥ) is a breathy release of the previous vowel, like a soft 'h' echo. Do not drop it.",
    "anusvara": "Anusvara (ṃ) nasalizes the preceding vowel. In chanting it often resonates through the nose gently.",
    "r_akar": "Syllabic ṛ (ऋ) is a vowel in Sanskrit, pronounced roughly 'ri' with short 'i' but vocalic r quality.",
    "jna": "jña (ज्ञ) is often realized as 'gya' or 'jnya' in North Indian styles; maintain clarity of both elements.",
    "sandhi": "Observe word-internal and phrase sandhi carefully. Sounds transform at boundaries (e.g. s + t → st, but also visarga changes).",
}

# Expected approximate phoneme sequence for Gayatri (used as reference for alignment hints)
# Each entry: (phoneme_or_syllable, expected_relative_duration, is_vowel, is_long, notes)
GAYATRI_PHONEME_SEQUENCE: List[Tuple[str, float, bool, bool, str]] = [
    ("oṃ", 1.8, True, True, "Long nasalized oṃ"),
    ("bhūr", 1.0, False, False, "Short u, aspirated bh"),
    ("bhu", 0.7, True, False, "Short u"),
    ("vaḥ", 1.1, True, False, "Visarga breath release"),
    ("tat", 0.85, False, False, "Dental t"),
    ("sa", 0.55, True, False, ""),
    ("vi", 0.55, True, False, ""),
    ("tur", 0.9, False, False, "Retroflex? usually dental here in chant"),
    ("va", 0.6, True, False, ""),
    ("re", 0.75, True, True, "Long e"),
    ("ṇyaṃ", 1.3, True, True, "Nasalized, ñ + y cluster"),
    ("bhar", 0.8, False, False, ""),
    ("go", 0.7, True, False, ""),
    ("de", 0.6, True, False, ""),
    ("va", 0.55, True, False, ""),
    ("sya", 0.7, False, False, "sya cluster"),
    ("dhī", 1.6, True, True, "Long ī - hold steady"),
    ("ma", 0.6, True, False, ""),
    ("hi", 0.7, True, False, ""),
    ("dhi", 0.65, True, False, ""),
    ("yo", 0.7, True, True, ""),
    ("yo", 0.65, True, False, ""),
    ("naḥ", 1.0, True, False, "Visarga"),
    ("pra", 0.7, False, False, ""),
    ("co", 0.65, True, False, ""),
    ("da", 0.6, True, False, ""),
    ("yāt", 1.4, True, True, "Long ā + final t with clean release"),
]

# =============================================================================
# WHISPER / ASR CONFIG
# =============================================================================
@dataclass
class ASRConfig:
    enabled: bool = False  # Default OFF because Sanskrit ASR is weak in stock models. Enable for experiments.
    model_size: str = "small"  # tiny, base, small, medium, large-v2, large-v3
    device: str = "cpu"  # or "cuda" if available
    compute_type: str = "int8"  # float16 on GPU recommended
    language: str = "sa"  # Sanskrit code per Whisper; falls back gracefully
    beam_size: int = 5


ASR_CONFIG = ASRConfig()


# =============================================================================
# VISUALIZATION
# =============================================================================
@dataclass
class VizConfig:
    dpi: int = 150
    waveform_figsize: Tuple[int, int] = (10, 3)
    pitch_figsize: Tuple[int, int] = (10, 3.5)
    spectrogram_figsize: Tuple[int, int] = (10, 4)


VIZ_CONFIG = VizConfig()


# =============================================================================
# FEEDBACK STYLE
# =============================================================================
@dataclass
class FeedbackConfig:
    show_phonetic_tips: bool = True
    max_tips_per_session: int = 3
    encouraging_tone: bool = True
    include_specific_examples: bool = True


FEEDBACK_CONFIG = FeedbackConfig()


def get_normalized_weights() -> Dict[str, float]:
    """Return a fresh copy of normalized weights."""
    total = sum(EVALUATION_WEIGHTS.values())
    return {k: v / total for k, v in EVALUATION_WEIGHTS.items()}


def get_weight(attribute: str) -> float:
    return get_normalized_weights().get(attribute, 0.0)
