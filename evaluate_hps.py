import os
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from app.app_manager import AppState
from app.analysis_engine import AnalysisEngine
from utils import helpers

K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
CENT_TOLERANCE = 200
EVAL_CONFIG = {"min_time_overlap": 0.3}
DATA_ROOT = "data"
RESULTS_DIR = "results/hps_k_variation"
os.makedirs(RESULTS_DIR, exist_ok=True)

app_state = AppState()
engine = AnalysisEngine(app_state)
results = []

for k in K_VALUES:
    print(f"\n========= Evaluating HPS with k = {k} =========\n")
    app_state.pitch_algorithm = "HPS"
    app_state.hps_k = k  # ← this is now supported

    for sample_dir in sorted(os.listdir(DATA_ROOT)):
        sample_path = os.path.join(DATA_ROOT, sample_dir)
        if not os.path.isdir(sample_path):
            continue

        wav_file = next((f for f in os.listdir(sample_path) if f.endswith(".wav")), None)
        if not wav_file:
            continue

        wav_path = os.path.join(sample_path, wav_file)
        f0_path = os.path.join(sample_path, "fundamental_ground_truth.csv")
        harm_path = os.path.join(sample_path, "harmonic_ground_truth.csv")
        if not (os.path.exists(f0_path) and os.path.exists(harm_path)):
            continue

        audio_data, _ = sf.read(wav_path)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        f0_df, harm_df = engine.load_ground_truth(f0_path, harm_path)
        ground_truth_df = pd.concat([
            f0_df.assign(label="fundamental"),
            harm_df.assign(label="harmonic")
        ], ignore_index=True)

        fund_times, fund_indices = [], []
        harm_times, harm_indices = [], []
        frame_size = engine.frame_length
        sr = engine.rate

        for i in range(0, len(audio_data) - frame_size, frame_size):
            chunk = audio_data[i:i + frame_size]
            result = engine.process_live_frame(chunk)
            pitch = result.get("pitch")
            t = i / sr
            if pitch:
                idx = helpers.freq_to_note_index(pitch)
                if idx is not None:
                    fund_times.append(t)
                    fund_indices.append(idx)
                h = result.get("active_harmonic")
                if h and h > 0:
                    h_idx = helpers.freq_to_note_index(pitch * h)
                    if h_idx is not None:
                        harm_times.append(t)
                        harm_indices.append(h_idx)

        detection_results = engine.format_detection_results(
            fund_times, fund_indices, harm_times, harm_indices,
            frame_duration=frame_size / sr
        )

        metrics = engine.evaluate_mode(
            mode="overtone_analyzer",
            f0_df=ground_truth_df,
            detection_results=detection_results,
            cent_tolerance=CENT_TOLERANCE,
            min_time_overlap=EVAL_CONFIG["min_time_overlap"]
        )

        results.append({
            "sample_name": sample_dir,
            "k": k,
            "fundamental_f1": round(metrics["fundamental"]["f1_score"], 4),
            "harmonic_f1": round(metrics["harmonic"]["f1_score"], 4),
            "combined_f1": round(metrics["global"]["f1_score"], 4)
        })

# === SAVE CSV AND PLOT ===
df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "hps_k_comparison.csv")
df.to_csv(csv_path, index=False, float_format="%.4f")

summary = df.groupby("k")[["fundamental_f1", "harmonic_f1", "combined_f1"]].mean().reset_index()
plt.figure(figsize=(8, 5))
plt.plot(summary["k"], summary["combined_f1"], marker="o", label="Combined F1")
plt.plot(summary["k"], summary["fundamental_f1"], marker="x", linestyle="--", label="Fundamental F1")
plt.plot(summary["k"], summary["harmonic_f1"], marker="s", linestyle="--", label="Harmonic F1")
plt.title("HPS Performance vs. k Parameter @ 200 cents")
plt.xlabel("k (number of harmonics in HPS)")
plt.ylabel("F1 Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "hps_k_vs_f1.png"))
plt.close()

print(f"✅ Results saved to {csv_path} and plot generated.")
