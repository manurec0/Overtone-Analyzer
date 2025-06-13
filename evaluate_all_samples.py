import os
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from app.app_manager import AppState
from app.analysis_engine import AnalysisEngine
from utils import helpers

# === CONFIGURATION ===
CENT_TOLERANCES = [20, 40, 60, 80, 100, 120, 150, 200, 300]
EVAL_CONFIG = {"min_time_overlap": 0.3}
DATA_ROOT = "data"
ALGORITHMS = ["SWIPE", "YIN", "pYIN", "HPS"]
OUTPUT_CSV = "evaluation_results_multi_tolerance.csv"
RESULTS_DIR = "results"

# === INIT APP STATE AND ENGINE ===
app_state = AppState()
engine = AnalysisEngine(app_state)

# === INIT FINAL RESULT CONTAINER ===
all_results = []

# === LOOP OVER CENT TOLERANCES ===
for cent_tol in CENT_TOLERANCES:
    print(f"\n================ Evaluating with CENT TOLERANCE: {cent_tol} cents ================\n")

    for sample_dir in sorted(os.listdir(DATA_ROOT)):
        sample_path = os.path.join(DATA_ROOT, sample_dir)
        if not os.path.isdir(sample_path):
            continue

        print(f"\n📂 Processing sample: {sample_dir}")

        wav_file = next((f for f in os.listdir(sample_path) if f.endswith(".wav")), None)
        if not wav_file:
            print("⚠️ No .wav file found, skipping.")
            continue

        wav_path = os.path.join(sample_path, wav_file)
        f0_path = os.path.join(sample_path, "fundamental_ground_truth.csv")
        harm_path = os.path.join(sample_path, "harmonic_ground_truth.csv")

        if not (os.path.exists(f0_path) and os.path.exists(harm_path)):
            print("⚠️ Missing ground truth files, skipping.")
            continue

        # Load audio
        audio_data, _ = sf.read(wav_path)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Load ground truth
        f0_df, harm_df = engine.load_ground_truth(f0_path, harm_path)
        ground_truth_df = pd.concat([
            f0_df.assign(label="fundamental"),
            harm_df.assign(label="harmonic")
        ], ignore_index=True)

        # Ensure sample-specific results directory exists
        sample_results_dir = os.path.join(RESULTS_DIR, sample_dir)
        os.makedirs(sample_results_dir, exist_ok=True)

        for algo in ALGORITHMS:
            print(f"🎯 Running {algo}...")
            app_state.pitch_algorithm = algo

            # Run pitch + overtone detection
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
                        harm_freq = pitch * h
                        h_idx = helpers.freq_to_note_index(harm_freq)
                        if h_idx is not None:
                            harm_times.append(t)
                            harm_indices.append(h_idx)

            detection_results = engine.format_detection_results(
                fund_times, fund_indices, harm_times, harm_indices,
                frame_duration=engine.frame_length / engine.rate
            )

            # Run evaluation
            metrics = engine.evaluate_mode(
                mode="overtone_analyzer",
                f0_df=ground_truth_df,
                detection_results=detection_results,
                cent_tolerance=cent_tol,
                min_time_overlap=EVAL_CONFIG["min_time_overlap"]
            )

            f_score = metrics["fundamental"]["f1_score"]
            h_score = metrics["harmonic"]["f1_score"]
            g_score = metrics["global"]["f1_score"]

            all_results.append({
                "sample_name": sample_dir,
                "cent_tolerance": cent_tol,
                "algorithm": algo,
                "fundamental_f1": round(f_score, 4),
                "harmonic_f1": round(h_score, 4),
                "combined_f1": round(g_score, 4)
            })

            # === GENERATE PLOT ===
            fig, ax = plt.subplots(figsize=(10, 4))
            for _, row in ground_truth_df.iterrows():
                color = "cyan" if row["label"] == "fundamental" else "green"
                ax.hlines(row["frequency"], row["time"], row["end_time"], color=color, linewidth=2)

            fund_times_plot = [r["start_time"] for r in detection_results if r["type"] == "fundamental"]
            fund_freqs_plot = [r["frequency"] for r in detection_results if r["type"] == "fundamental"]
            harm_times_plot = [r["start_time"] for r in detection_results if r["type"] == "harmonic"]
            harm_freqs_plot = [r["frequency"] for r in detection_results if r["type"] == "harmonic"]

            ax.scatter(fund_times_plot, fund_freqs_plot, color="blue", s=10, label="Detected Fundamental")
            ax.scatter(harm_times_plot, harm_freqs_plot, color="red", s=10, label="Detected Harmonic")

            ax.set_title(f"{sample_dir} - {algo} @ {cent_tol}c")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()

            plot_filename = f"{algo}_{cent_tol}c.png"
            plot_path = os.path.join(sample_results_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close(fig)

# === SAVE COMBINED CSV WITH FORMATTED FLOATS ===
df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_CSV, index=False, float_format="%.4f")
print(f"\n✅ All evaluations and plots saved. CSV: {OUTPUT_CSV}")

# === PLOT ALL F1 CURVES VS CENT TOLERANCE ===
print("📊 Generating tolerance vs. score plots...")

# Reload dataframe to ensure it's clean
df = pd.read_csv(OUTPUT_CSV)

# Create summary plot directory
summary_plot_dir = os.path.join(RESULTS_DIR, "summary_plots")
os.makedirs(summary_plot_dir, exist_ok=True)

# Define which F1 types to plot
f1_types = {
    "combined_f1": "Combined",
    "fundamental_f1": "Fundamental",
    "harmonic_f1": "Harmonic"
}

# 1. Plot ALL algorithms per F1 type (1 plot per score type)
for f1_col, label in f1_types.items():
    plt.figure(figsize=(8, 5))
    for algo in ALGORITHMS:
        algo_df = df[df['algorithm'] == algo]
        grouped = algo_df.groupby('cent_tolerance')[f1_col]
        means = grouped.mean()
        stds = grouped.std()
        plt.plot(means.index, means.values, marker='o', label=algo)
        plt.fill_between(means.index, means - stds, means + stds, alpha=0.2)
    plt.title(f"Average {label} F1-Score vs Cent Tolerance")
    plt.xlabel("Cent Tolerance")
    plt.ylabel(f"{label} F1-Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fname = f"tolerance_{f1_col}_vs_tolerance.png"
    plt.savefig(os.path.join(summary_plot_dir, fname))
    plt.close()

# 2. Plot individual curves per algorithm and F1 type (1 plot per algo & score type)
for algo in ALGORITHMS:
    algo_df = df[df['algorithm'] == algo]
    for f1_col, label in f1_types.items():
        grouped = algo_df.groupby('cent_tolerance')[f1_col]
        means = grouped.mean()
        stds = grouped.std()
        plt.figure(figsize=(8, 5))
        plt.plot(means.index, means.values, marker='o', label=algo)
        plt.fill_between(means.index, means - stds, means + stds, alpha=0.2)
        plt.title(f"{algo} - {label} F1-Score vs Cent Tolerance")
        plt.xlabel("Cent Tolerance")
        plt.ylabel(f"{label} F1-Score")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fname = f"{algo}_{f1_col}_vs_tolerance.png"
        plt.savefig(os.path.join(summary_plot_dir, fname))
        plt.close()

print(f"✅ All tolerance curve plots saved to: {summary_plot_dir}")
