import numpy as np

# Frame-level melody matching: no note segmentation, so onset errors don't break the match.
# Query = YIN pitch contour; target = simplified notation expanded by beats into a step contour.

DOWNSAMPLE = 3                          # YIN frames per DTW frame (hop 256 @ 8 kHz -> ~96 ms)
TEMPO_SCALES = (0.5, 0.7, 1.0, 1.4, 2.0)  # global tempo ratios tried (linear scaling)
KEY_SHIFTS = np.arange(-6, 7)           # semitones searched around the median-aligned key
COST_CLIP = 3.0                         # per-frame cost cap (semitones), limits damage from bad frames


def query_contour(f0, times, onsets_sec):
    """Voiced YIN frames between first and last onset -> MIDI (float), 3-pt median, then block-mean downsampled."""
    keep = ~np.isnan(f0)
    if len(onsets_sec) >= 2:
        keep &= (times >= onsets_sec[0]) & (times <= onsets_sec[-1])
    midi = 69 + 12 * np.log2(f0[keep] / 440.0)
    if len(midi) < DOWNSAMPLE:
        return midi
    midi = np.array([np.median(midi[max(0, i - 1):i + 2]) for i in range(len(midi))])
    return midi[: len(midi) // DOWNSAMPLE * DOWNSAMPLE].reshape(-1, DOWNSAMPLE).mean(axis=1)


def target_contour(midi_numbers, beats, frames_per_beat):
    reps = np.maximum(1, np.round(np.asarray(beats, float) * frames_per_beat)).astype(int)
    return np.repeat(np.asarray(midi_numbers, float), reps)


def subsequence_dtw(q, t, shifts):
    """
    DTW of the whole query against any segment of the target (free start/end).
    Steps (1,1), (1,2), (2,1): local tempo 0.5x-2x, and each row depends only on the two previous
    rows, so a row is computed as one numpy operation. Each key shift keeps its own path.
    Returns the mean per-frame cost of the best path.
    """
    INF = 1e18
    def cost(i):
        return np.minimum(np.abs(q[i] - shifts[:, None] - t[None, :]), COST_CLIP)
    prev2 = np.full((len(shifts), len(t)), INF)
    prev1 = cost(0)
    for i in range(1, len(q)):
        cur = np.full_like(prev1, INF)
        cur[:, 1:] = np.minimum(prev1[:, :-1], prev2[:, :-1])
        cur[:, 2:] = np.minimum(cur[:, 2:], prev1[:, :-2])
        cur += cost(i)
        prev2, prev1 = prev1, cur
    return prev1.min() / len(q)


def dtw_distance(q, midi_numbers, beats, note_sec, frame_sec):
    """
    q: query contour from query_contour(); note_sec: median detected note length of the query (s);
    frame_sec: duration of one contour frame (YIN hop / rate * DOWNSAMPLE).
    Target tempo is first matched to the query's note rate, then TEMPO_SCALES are tried around it.
    """
    if len(q) < 3:
        return np.inf
    base_fpb = (note_sec / frame_sec) / np.median(beats)
    best = np.inf
    for scale in TEMPO_SCALES:
        t = target_contour(midi_numbers, beats, base_fpb * scale)
        shifts = np.round(np.median(q) - np.median(t)) + KEY_SHIFTS
        best = min(best, subsequence_dtw(q, t, shifts.astype(float)))
    return best
