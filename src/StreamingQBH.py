import os
import sys
import time
import pickle
import numpy as np

import OnsetDetection
import main

# Streaming QBH: run the query on the audio received so far every STEP_SEC seconds,
# turn the fused scores into a softmax confidence, and answer as soon as the top song is confident enough.

STEP_SEC = 2.5    # run QBH every 2.5 s of received audio
START_SEC = 5.0   # first run once 5 s have arrived
MIN_LAST_SEC = 0.5  # skip a final batch that would add less audio than this
TEMPERATURE = 0.30  # softmax temperature, calibrated by calibrate_temperature() on the 10/15/20 s sets
CONF_THRESHOLD = 0.95  # answer early once the top song reaches this; no accuracy loss on any set (see report())


def batch_times(duration_sec, step=STEP_SEC, start=START_SEC):
    """Prefix lengths to evaluate: start, start+step, ..., always ending with the full recording."""
    times = list(np.arange(start, duration_sec, step))
    if times and duration_sec - times[-1] < MIN_LAST_SEC:
        times.pop()
    return times + [duration_sec]


def confidence(score, temperature):
    """Softmax over candidates of -score / T (lower fused score = more similar). Pruned songs have probability 0."""
    z = -np.asarray(score, dtype=float) / temperature
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def stream_query(data, rate, targets, transition_matrices, step=STEP_SEC, start=START_SEC):
    """
    Run QBH on each growing prefix of the recording.
    Returns a list of batches: dict(t, candidates, score, elapsed). candidates is empty when the prefix
    has too few notes to compare (that batch cannot produce an answer).
    """
    history = []
    for t in batch_times(len(data) / rate, step, start):
        t0 = time.perf_counter()
        features = main.extract_query_features(data[: int(round(t * rate))], rate)
        if len(features[0]) < 2:  # fewer than 3 notes: nothing to match yet
            candidates, score = [], np.array([])
        else:
            candidates, score = main.score_candidates(features, targets, transition_matrices)
        history.append(dict(t=t, candidates=candidates, score=np.asarray(score), elapsed=time.perf_counter() - t0))
    return history


def first_confident(history, temperature=TEMPERATURE, threshold=CONF_THRESHOLD):
    """
    Online decision: the first batch whose top song reaches `threshold`; otherwise the full-length batch.
    Returns (song, confidence, t, early) where early = answered before the recording ended.
    """
    for i, batch in enumerate(history):
        if not batch['candidates']:
            continue
        p = confidence(batch['score'], temperature)
        k = int(np.argmax(p))
        if p[k] >= threshold or i == len(history) - 1:
            return batch['candidates'][k], float(p[k]), batch['t'], i < len(history) - 1
    return None, 0.0, history[-1]['t'], False


def calibrate_temperature(records, grid=np.linspace(0.05, 2.0, 40)):
    """Temperature minimising the mean -log P(correct song) on the full-length batches."""
    def nll(T):
        losses = []
        for label, history in records:
            last = history[-1]
            if not last['candidates']:
                continue
            p = confidence(last['score'], T)
            p_true = p[last['candidates'].index(label)] if label in last['candidates'] else 0.0
            losses.append(-np.log(max(p_true, 1e-12)))
        return np.mean(losses)
    return float(min(grid, key=nll))


def collect(query_dir, target_file, transition_matrix_folder):
    """Run every batch of every query in query_dir (no early stop) so thresholds can be swept offline."""
    targets = main.load_target_data(target_file)
    transition_matrices = main.load_hmms(targets, transition_matrix_folder)
    records = []
    for audio_file in sorted(os.listdir(query_dir)):
        if not audio_file.endswith('.wav'):
            continue
        print(f"Processing {audio_file}...", flush=True)
        data, rate, _ = OnsetDetection.load_audio(os.path.join(query_dir, audio_file))
        label = os.path.splitext(audio_file)[0].split("_")[1]
        records.append((label, stream_query(data, rate, targets, transition_matrices)))
    return records


def report(datasets, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
    """datasets: {name: records}. Temperature is calibrated once on all records together."""
    all_records = [r for records in datasets.values() for r in records]
    T = calibrate_temperature(all_records)
    print(f"Temperature (all data): {T:.2f}  |  per dataset: "
          + ", ".join(f"{name}: {calibrate_temperature(rs):.2f}" for name, rs in datasets.items()))

    for name, records in datasets.items():
        n = len(records)
        full_acc = np.mean([h[-1]['candidates'][int(np.argmin(h[-1]['score']))] == label
                            if h[-1]['candidates'] else False for label, h in records])
        elapsed = [b['elapsed'] for _, h in records for b in h]
        durations = [h[-1]['t'] for _, h in records]
        print(f"\n=== {name}: {n} queries, mean length {np.mean(durations):.1f} s, full-length accuracy {full_acc:.1%}, "
              f"compute per batch mean {np.mean(elapsed):.2f} s / max {np.max(elapsed):.2f} s ===")

        # accuracy if we always answered at a fixed prefix length
        fixed = {}
        for label, h in records:
            for b in h[:-1]:
                if b['candidates']:
                    fixed.setdefault(b['t'], []).append(b['candidates'][int(np.argmin(b['score']))] == label)
        print("fixed prefix: " + ", ".join(f"{t:g}s {np.mean(v):.1%} (n={len(v)})" for t, v in sorted(fixed.items())))

        print(f"{'threshold':>9} {'accuracy':>9} {'early %':>8} {'mean t (s)':>11} {'saved (s)':>10} {'early acc':>10}")
        for th in thresholds:
            decisions = [(first_confident(h, T, th), label, h[-1]['t']) for label, h in records]
            correct = [d[0] == label for d, label, _ in decisions]
            early = [d[3] for d, _, _ in decisions]
            t_dec = [d[2] for d, _, _ in decisions]
            saved = [full - d[2] for d, _, full in decisions]
            early_correct = [c for c, e in zip(correct, early) if e]
            early_acc = f"{np.mean(early_correct):.1%}" if early_correct else "-"
            print(f"{th:>9.2f} {np.mean(correct):>9.1%} {np.mean(early):>8.1%} {np.mean(t_dec):>11.2f} "
                  f"{np.mean(saved):>10.2f} {early_acc:>10}")


if __name__ == "__main__":
    # python StreamingQBH.py collect hummingdata/20 out_20.pkl   -> run all batches, save
    # python StreamingQBH.py report out_10.pkl out_15.pkl out_20.pkl -> calibrate + threshold sweep
    transition_matrix_folder = r"Project_in_Lin's_thesis/HMM_midi_diff"
    target_file = "hummingdata/Target_tempo_50_utf-8.txt"

    if sys.argv[1] == "collect":
        records = collect(sys.argv[2], target_file, transition_matrix_folder)
        with open(sys.argv[3], 'wb') as f:
            pickle.dump(records, f)
    elif sys.argv[1] == "report":
        datasets = {}
        for path in sys.argv[2:]:
            with open(path, 'rb') as f:
                datasets[os.path.splitext(os.path.basename(path))[0]] = pickle.load(f)
        report(datasets)
