# SanskritMantraPronunciationCoach

A complete, production-quality Python tool for learning **authentic Sanskrit mantra pronunciation** with precise, multi-dimensional feedback.

Built with deep respect for Sanskrit phonetics (mātrā/vowel length, aspiration, retroflexion, visarga, anusvara, and prosodic stability).

---

## ✨ Features

- **Mantra Database** — JSON definitions with Devanagari, IAST, translation, phonetic guidance, and common pitfalls.
- **Reference Analysis** — High-quality prosodic extraction using Praat (via parselmouth) for pitch, energy, voicing.
- **Live Recording** — Clean microphone capture with pyaudio + visual RMS feedback.
- **8 Configurable Evaluation Dimensions** (fully tunable in `config.py`):
  1. Pronunciation Accuracy (acoustic + optional ASR)
  2. Vowel Length Accuracy (**mātrā** — critically important in Sanskrit)
  3. Tone / Pitch Stability
  4. Speed / Tempo + rhythm
  5. Smoothness / Fluency
  6. Volume Consistency
  7. Breath Control
  8. Overall Similarity (DTW-based)
- **Rich CLI Feedback** — Beautiful tables, color-coded scores, Sanskrit-specific tips.
- **Diagnostic Visualizations** — Waveform, pitch contour, and energy envelope comparisons saved as PNGs.
- **Synthetic Reference Bootstrap** — Works out-of-the-box even without real recordings (real references are strongly recommended).
- **Extensible** — Easy to add new mantras.

---

## 📁 Project Structure

The project is intentionally flat at the repository root for simplicity:

```
.
├── main.py                 # Typer CLI entrypoint
├── __main__.py             # Allows `python -m sanskrit_mantra_coach`
├── config.py               # All weights, thresholds, Sanskrit phoneme inventory, ASR settings
├── requirements.txt
├── mantras/
│   └── gayatri_mantra.json # Full Gayatri definition + phonetic guidance
├── audio/
│   ├── references/         # Gold-standard recordings (auto-generates synthetic if missing)
│   └── user_recordings/    # Your practice attempts + timestamped reports
├── core/
│   ├── audio_processor.py  # Load/save/record/play + MFCC, pitch (parselmouth), energy, DTW
│   ├── phoneme_aligner.py  # Devanagari/IAST handling, vowel detection, alignment quality
│   ├── evaluator.py        # All 8 scoring dimensions with detailed comments
│   └── feedback_generator.py
├── utils/
│   ├── helpers.py
│   └── visualizations.py   # Matplotlib comparison plots
├── reports/                # Saved JSON evaluation reports
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Install System Dependencies

**macOS** (most common for this workspace):
```bash
brew install portaudio   # required for pyaudio
```

**Linux (Debian/Ubuntu)**:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

### 2. Python Environment

**Requirements:** Python 3.10 or newer (Python 3.12+ or 3.14 recommended).

```bash
# Recommended on macOS with Homebrew
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

**Important after upgrading Python** (e.g. from 3.9 → 3.14):
Old virtual environments are tied to the Python version they were created with and will break. Delete and recreate the venv:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note on faster-whisper**: This pulls in torch (~2GB). If you only want acoustic/prosodic evaluation (recommended for Sanskrit), leave `ASR_CONFIG.enabled = False` in `config.py` (the default). The system works excellently without it.

### 3. First Run

```bash
python -m main list-mantras
```

The first time you practice, a **synthetic reference** will be auto-generated for Gayatri Mantra so you can immediately test the full pipeline.

---

## 🎵 Recommended Audio Formats

For the most accurate pronunciation coaching results, audio format matters:

| Format     | For Reference Files | For User Recordings | Notes |
|------------|---------------------|---------------------|-------|
| **WAV**    | ★★★★★ (Recommended) | ★★★★★ (Default)     | Lossless, precise timing. Best choice. |
| **FLAC**   | ★★★★☆               | ★★★★☆               | Lossless + compressed. Excellent. |
| **OGG**    | ★★☆☆☆               | ★★★☆☆               | Lossy. Usable but avoid for references. |
| **MP3**    | ★☆☆☆☆               | ★★☆☆☆               | Lossy + variable quality. Not recommended. |
| **MOV / MP4** | —                | —                   | Video containers. Supported only if **ffmpeg** is installed (audio is extracted automatically). |

**Strong recommendation:**
- Store your **reference mantras** as **WAV** or **FLAC**.
- The app automatically warns you when you load lossy formats for evaluation.
- Video files (`.mov`, `.mp4`, `.mkv`, etc.) are supported **if ffmpeg is installed**. The tool will extract the audio track on the fly.

Install ffmpeg (highly recommended):
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

---

## 🎤 Core Commands

### List mantras
```bash
python -m main list-mantras
```

### Full Practice Session (Recommended)
```bash
python -m main practice gayatri_mantra
```

This will:
1. Display the full Gayatri in Devanagari + IAST + meaning.
2. Optionally play the reference.
3. Prompt you to record (press Enter → Ctrl-C to stop).
4. Run the full 8-dimension analysis.
5. Show gorgeous rich feedback + Sanskrit phonetic tips.
6. Generate 3 diagnostic plots (waveform / pitch / energy).
7. Save a timestamped JSON report + your WAV.

### Evaluate an Existing Recording
```bash
python -m main evaluate gayatri_mantra --audio audio/user_recordings/gayatri_mantra_user_....wav
```

### Record a High-Quality Reference (Teachers / Advanced)
```bash
python -m main record-reference gayatri_mantra
```

This replaces the reference file. All future evaluations use it as the gold standard.

### Show Detailed Mantra Info
```bash
python -m main info gayatri_mantra
```

---

## ⚙️ Configuration & Tuning

Everything important lives in [config.py](config.py):

- **Weights** (`EVALUATION_WEIGHTS`) — Change emphasis (e.g. give `vowel_length_accuracy` even higher weight).
- **Thresholds** — Adjust what counts as "Excellent" vs "Needs Work".
- **Metric-specific tuning** — `METRIC_CONFIG` controls vowel length tolerance, breath detection windows, DTW cutoffs, etc.
- **Sanskrit Phoneme Inventory** — Extend `SANSKRIT_CONSONANTS`, `SANSKRIT_VOWELS`, and guidance strings.
- **ASR** — Enable/disable Whisper, change model size (tiny/base/small recommended for speed).

After editing `config.py`, just re-run any command — no reinstall needed.

---

## 🧠 Sanskrit Phonetic Priorities

The system is deliberately biased toward dimensions that matter most for traditional Vedic/Sanskrit chanting:

| Dimension                  | Why It Matters                                      | Common Learner Error                     |
|---------------------------|-----------------------------------------------------|------------------------------------------|
| Vowel Length (mātrā)      | Changes meaning and meter                           | Shortening ī in *dhīmahi*                |
| Retroflex vs Dental       | ṭ/ḍ/ṇ/ṣ vs t/d/n                                    | Using English "t" for ट                  |
| Aspiration                | kh/gh/th/dh/ph/bh have clear breath release         | Pronouncing like unaspirated English     |
| Visarga (ḥ)               | Breathy release, not a hard stop                    | Dropping it entirely                     |
| Anusvara (ṃ)              | Gentle nasal resonance                              | Over-closing or turning into "m"         |
| Pitch Stability           | Traditional chanting favors steady, meditative tone | Wobble or scooping on long vowels        |
| Breath Placement          | Natural, unobtrusive pauses between pādas           | Gasping mid-phrase or no breathing       |

---

## 📊 Understanding the Scores

Each attribute returns:
- **Score (0–100)**
- **Weighted contribution** (according to `config.py`)
- **Specific, actionable comment**

The **Overall Score** is the weighted sum (always shown with qualitative label).

Visualizations (saved automatically) let you *see*:
- Where your timing diverged (waveform)
- Where pitch wobbled or drifted (pitch contour)
- Where volume or phrasing was uneven (energy envelope)

---

## 🗂️ Adding a New Mantra

1. Create `mantras/your_mantra.json` modeled exactly after `gayatri_mantra.json`.
2. Record or generate a clean reference → place in `audio/references/your_mantra.wav` (or use `record-reference` command).
3. (Optional) Provide a detailed `phoneme_sequence` with relative durations for better vowel-length scoring.
4. Test with `practice your_mantra`.

The more accurate and traditionally rendered your reference recording, the better the feedback quality.

---

## ⚠️ Limitations & Honest Notes

- **ASR for Sanskrit is weak** in stock Whisper models. The system therefore relies primarily on **acoustic + prosodic similarity** (DTW, pitch, energy, vowel segmentation). This is actually more reliable for pronunciation coaching than text in this domain.
- Synthetic references are useful for pipeline testing but **not** for serious practice. Always replace with a high-quality human recording (ideally from a qualified teacher).
- Real forced alignment at the phoneme level would require a custom acoustic model trained on Sanskrit chanting. The current implementation uses DTW + heuristics — excellent for feedback, not perfect.
- Works best with clear, close-mic recordings in a quiet space.

---

## 📚 Recommended Real References

For serious use, source clean recordings of:
- Gayatri Mantra by respected Vedic chanters (e.g. traditional South Indian or North Indian pāṭhaśālā styles)
- Maintain consistent pitch center and traditional Gāyatrī rhythm

---

## 🛠️ Development & Contributing

- All magic numbers and Sanskrit-specific rules are in `config.py` — start there.
- Core evaluation logic is deliberately separated from UI (easy to build a web or mobile frontend).
- JSON reports in `reports/` are designed to be usable as a dataset for future ML work (goodness-of-pronunciation modeling, etc.).

---

## 🙏 Acknowledgments

This project exists because authentic Sanskrit pronunciation is a living transmission. The goal is not perfection on first try, but **mindful, informed practice** that respects the phonetic precision the ṛṣis embedded in the language.

May your chanting be steady, clear, and joyful.

---

**License**: MIT (feel free to adapt for your own saṅgha, āśrama, or personal sādhana).

**Author**: Built as a complete, self-contained expert-level demonstration of speech processing + Sanskrit pedagogy in Python.
