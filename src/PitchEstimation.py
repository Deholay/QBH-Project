import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import convolve

def create_smoother(L):
    # Generate the smoother array based on L
    smoother = np.array([(L - abs(m)) / L**2 if abs(m) < L else 0 for m in range(-L, L+1)])
    return smoother

def fft_between_onsets(data, rate, onsets, time_slot_width, L=10, PLOT_PITCH = True):
    # Generate the smoother based on L
    smoother = create_smoother(L)
    
    # List to store FFT results and detected pitches for each segment
    fft_results = []
    pitch_results = []

    for i in range(len(onsets) - 1):
        # Get the start and end indices for the segment between onset[i] and onset[i+1]
        start_sample = int(onsets[i] * time_slot_width * rate)
        end_sample = int(onsets[i+1] * time_slot_width * rate)
        
        # Extract the segment of the signal
        segment = data[start_sample:end_sample]

        # Perform FFT on the segment
        fft_result = fft(segment)

        # Compute the frequency axis for this segment
        freq_axis = np.fft.fftfreq(len(segment), 1/rate)

        # Smooth the FFT result by convolving with the smoother
        smoothed_fft_result = convolve(np.abs(fft_result), smoother, mode='same')

        # Find the fundamental frequency f0 based on conditions (1)-(4)
        max_val = np.max(smoothed_fft_result)
        candidate_m = [
            m for m in range(len(smoothed_fft_result)//2)  # Only consider up to Nyquist frequency
            if smoothed_fft_result[m] > smoothed_fft_result[m+1]  # Condition (1)
            and 80 < freq_axis[m] < rate / 2  # Condition (2)
            and smoothed_fft_result[m] > max_val * 0.2  # Condition (3)
        ]

        # (4) Choose the smallest m that satisfies the conditions
        if candidate_m:
            m0 = min(candidate_m)
            f0 = freq_axis[m0]  # Fundamental frequency
        else:
            f0 = None  # No valid pitch found for this segment

        # Store the pitch and smoothed FFT result for plotting and analysis
        fft_results.append((freq_axis, smoothed_fft_result))
        pitch_results.append(f0)

        if(PLOT_PITCH == True):
            # Plot the smoothed FFT result for this segment
            plt.figure(figsize=(10, 5))
            plt.plot(freq_axis[:len(freq_axis)//2], smoothed_fft_result[:len(smoothed_fft_result)//2])
            plt.title(f'Smoothed FFT of Segment between Onset {i} and Onset {i+1}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            if f0:
                plt.axvline(x=f0, color='red', linestyle='--', label=f'Pitch f0 = {f0:.2f} Hz')
                plt.legend()
            plt.show()

    return fft_results, pitch_results

def yin_f0(x, rate, frame=2048, hop=256, threshold=0.15, fmin=70.0, fmax=1000.0):
    """
    YIN (de Cheveigné & Kawahara, 2002) frame-level f0.
    Returns (f0, times): f0 in Hz per frame (np.nan = unvoiced), times = frame centre in seconds.
    """
    x = x.astype(float)
    x = x / (np.max(np.abs(x)) + 1e-9)
    if len(x) < frame:
        return np.array([]), np.array([])
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    W = frame // 2
    tau_min, tau_max = int(rate / fmax), min(int(rate / fmin), W - 1)

    # Difference function d(tau) = sum_{j<W} (x_j - x_{j+tau})^2, via FFT cross-correlation
    acf = np.fft.irfft(np.fft.rfft(frames, 2 * frame) * np.conj(np.fft.rfft(frames[:, :W], 2 * frame)), 2 * frame)[:, :W]
    energy = np.cumsum(np.concatenate([np.zeros((len(frames), 1)), frames ** 2], axis=1), axis=1)
    taus = np.arange(W)
    diff = energy[:, W][:, None] + (energy[:, taus + W] - energy[:, taus]) - 2 * acf

    # Cumulative mean normalized difference
    cmnd = np.ones_like(diff)
    cmnd[:, 1:] = diff[:, 1:] * taus[1:] / np.maximum(np.cumsum(diff[:, 1:], axis=1), 1e-12)

    f0 = np.full(len(frames), np.nan)
    for i, c in enumerate(cmnd):
        below = np.where(c[tau_min:tau_max] < threshold)[0]
        if len(below) == 0:
            continue  # unvoiced
        t = below[0] + tau_min
        while t + 1 < tau_max and c[t + 1] < c[t]:  # walk to the local minimum
            t += 1
        y0, y1, y2 = c[t - 1], c[t], c[t + 1]  # parabolic interpolation
        denom = y0 - 2 * y1 + y2
        f0[i] = rate / (t + 0.5 * (y0 - y2) / denom if denom else t)
    times = (np.arange(len(frames)) * hop + frame / 2) / rate
    return f0, times

def yin_between_onsets(f0, times, onsets, time_slot_width):
    """
    Pitch of each note = median of YIN frame f0 (from yin_f0) between consecutive onsets (None if no voiced frame).
    Replaces fft_between_onsets: its "lowest peak above 20% of max" rule often picked noise / sub-harmonics,
    producing single-note pitch spikes.
    """
    pitch_results = []
    for i in range(len(onsets) - 1):
        start, end = onsets[i] * time_slot_width, onsets[i + 1] * time_slot_width
        voiced = f0[(times >= start) & (times < end) & ~np.isnan(f0)]
        pitch_results.append(float(np.median(voiced)) if len(voiced) else None)
    return pitch_results

# Convert pitch to MIDI and calculate MIDI differences
def calculate_midi_differences(pitch_results):
    # Convert each fundamental frequency to a MIDI note number
    midi_numbers = np.ceil([48 + 12 * np.log2(f0 / 261.63) for f0 in pitch_results if f0])      #"Proposed Algorithm" 直接無條件進位
    
    # Calculate the differences between consecutive MIDI numbers
    midi_differences = np.diff(midi_numbers)

    return midi_numbers, midi_differences

# Calculate beat intervals
def calculate_beat_intervals(onsets, time_slot_width):
    # Calculate intervals between consecutive onsets
    intervals = np.diff(onsets) * time_slot_width  # Convert onset intervals to seconds
    
    # Find the median interval (b0)
    b0 = np.median(intervals)
    
    # Calculate beat for each interval
    beats = [2 ** np.round(np.log2(interval / b0)) for interval in intervals]
    
    return beats