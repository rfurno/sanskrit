You are an expert Python developer and AI/ML engineer specializing in speech processing. Build a complete, well-structured Python project called **SanskritMantraPronunciationCoach**.

### Project Goal
Create a tool that:
1. Learns the correct pronunciation of Sanskrit mantras using reference audio files + Devanagari text.
2. Records a user reciting the mantra.
3. Evaluates the user's repetition across multiple configurable dimensions and gives detailed feedback.

### Core Features

**1. Mantra Database**
- Store mantras with Devanagari text and corresponding reference audio.

**2. Reference Analysis**
- Analyze reference audio to extract pronunciation features.

**3. User Evaluation**
- Record user audio through microphone.
- Compare user audio with reference and generate scores.

### Configurable Evaluation Attributes
Make these fully configurable via a `config.yaml` or `config.py` file:

- **Pronunciation Accuracy** (phoneme-level match, especially Sanskrit-specific sounds)
- **Vowel Length Accuracy** (short vs long vowels - critical in Sanskrit)
- **Tone / Pitch Stability**
- **Speed / Tempo** (duration and rhythm match)
- **Smoothness / Fluency** (pauses, hesitations, flow)
- **Volume Consistency**
- **Breath Control** (unnatural breathing gaps)
- **Overall Similarity Score**

Each attribute should return:
- Score (0-100)
- Weighted contribution
- Specific feedback comments

### Technical Stack (include requirements.txt)
- `faster-whisper` or `whisper` (for transcription)
- `librosa` + `soundfile` (audio processing)
- `praat-parselmouth` (prosody analysis)
- `pyaudio` (recording)
- `phonemizer` or `epitran` (Devanagari → phonemes)
- `numpy`, `scipy`, `jiwer`
- `rich` + `typer` (CLI interface)
- `pyyaml` (configuration)

### Recommended Project Structure
```bash
sanskrit_mantra_coach/
├── main.py                 # Entry point (CLI)
├── config.py               # Configuration and weights
├── requirements.txt
├── mantras/                # Database of mantras
│   ├── gayatri_mantra.json
│   └── ... 
├── audio/
│   ├── references/         # Original mantra recordings
│   └── user_recordings/    # Saved user attempts
├── core/
│   ├── audio_processor.py
│   ├── phoneme_aligner.py
│   ├── evaluator.py
│   └── feedback_generator.py
├── utils/
│   ├── helpers.py
│   └── visualizations.py
└── README.md
```

## Deliverables
Please generate:

Complete project structure with all files and their full code.
requirements.txt
config.py with adjustable weights for each evaluation attribute.
Sample mantra (Gayatri Mantra recommended).
Clear setup & usage instructions.
CLI commands: --list-mantras, --practice <mantra>, --evaluate <mantra>.

Make the code clean, well-documented, modular, and focused on Sanskrit phonetic accuracy. Prioritize proper handling of Devanagari and Sanskrit-specific sounds (aspirated consonants, retroflexes, vowel lengths, etc.).