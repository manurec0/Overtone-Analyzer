# Overtone Analyzer

**Overtone Analyzer** is the official repository accompanying the final-year thesis:  
**Development of a Real-Time Audio Analysis and Visualization Tool for Overtone Singing Using Fundamental Frequency Pitch Detection and Harmonic Extraction Techniques**.

This project presents a real-time application for analyzing and visualizing overtone singing. It integrates robust pitch estimation algorithms (SWIPE, YIN, HPS, pYIN), spectral harmonic extraction, and a modular GUI system for visualization and evaluation. The tool is designed for researchers, educators, and vocal practitioners interested in the detailed study of overtone singing.

---

## Dataset

### Overview

The dataset includes **15 overtone singing performances**, each sourced from YouTube and manually annotated. Annotations are provided in Sonic Visualiser-compatible CSV format and include both **fundamental frequency tracks** and, in select cases, **harmonic annotations**. The dataset supports evaluation of pitch detection methods under the unique spectral characteristics of overtone singing.

### Dataset Curation Process

- **Manual Selection**: Recordings were chosen based on clarity of overtone technique and stylistic variety (e.g., khoomei, kargyraa, harmonic singing).
- **Annotation**: Ground truth files were manually labeled for fundamental pitch and overtone content using Sonic Visualiser.
- **Evaluation-Ready**: Each file is aligned for algorithmic comparison using automated metrics (precision, recall, F1-score).

### Dataset Details

Each sample lies under its own folder in the `data/` directory and contains:
- `sample_name.wav`: the recorded performance
- `fundamental_ground_truth.csv`: fundamental frequency annotation
- `harmonic_ground_truth.csv`: harmonic annotations (optional)

---

## Visualization Tool

### Overview

The GUI-based application provides multiple modes for real-time and offline audio analysis:

- **Waveform Mode**: Displays time-domain amplitude with scrolling playhead
- **Piano Roll Mode**: Shows detected pitch over time as note blocks
- **Spectrogram Mode**: Log-scaled frequency heatmap with optional zoom
- **Overtone Profile Mode**: Extracts harmonics and labels intervals
- **Oscilloscope Mode**: Displays waveform at a per-frame level during playback

### Features

- Live microphone or WAV file input
- Real-time pitch tracking using SWIPE, YIN, HPS, or pYIN
- Custom ground truth overlay and visual evaluation
- Modular UI built with PySide6 and PyQtGraph
- Internal evaluation engine for precision, recall, and F1 metrics

---

## Installation

We recommend using a Python virtual environment:

```bash
git clone https://github.com/yourusername/OvertoneAnalyzer.git
cd OvertoneAnalyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Running the Application

To launch the visualization tool:

```bash
python main.py
```

