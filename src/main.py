import OnsetDetection
import PitchEstimation
import MelodyMatching
import DP
import HMM
import os
import numpy as np

# Load the target data
def load_target_data(file_path):
    """
    Load target data and convert the simplified notation to MIDI differences.
    """
    targets = {}
    with open(file_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig 去除檔頭 BOM，避免第一首歌名讀成 BOM+三輪車
        for line in f:
            parts = line.strip().split()  # Split into song name, simplified notation, and beats
            song_name = parts[0]
            simplified_notation = parts[1].split('/')[0]
            target_beat = [int(digit) for digit in parts[1].split('/')[1]]
            _, target_diff = MelodyMatching.simplified_notation_to_midi(simplified_notation)
            targets[song_name] = [target_diff, target_beat]
    return targets

# Extract MIDI differences from audio files
def process_audio_to_query_diff(audio_file):
    """
    Process audio to extract MIDI differences.
    """
    
    time_slot_width = 0.01
    rho = 0.02
    laMbda = 0.7

    data, rate, _ = OnsetDetection.load_audio(audio_file)
    onsets, _ = OnsetDetection.Onset_Detection(data, rate, time_slot_width, rho, laMbda, k=3.0, use_median=True)
    onsets = OnsetDetection.refine_onsets(onsets, data, rate, time_slot_width, min_interval=0.15, start_offset=0.2, end_offset=0.2, PLOT_ONSET = False)

    f0, f0_times = PitchEstimation.yin_f0(data, rate)
    pitch_results = PitchEstimation.yin_between_onsets(f0, f0_times, onsets, time_slot_width)
    _, query_diff = PitchEstimation.calculate_midi_differences(pitch_results)
    query_beat_raw = PitchEstimation.calculate_beat_intervals(onsets, time_slot_width)
    
    # 簡化 beat 處理 - 可能需要根據您的 target_beat 意義調整
    # 假設 target_beat 代表每個音符的節拍，長度等於音符數
    query_beat = []
    if query_beat_raw: # 如果至少有兩個音符（一個間隔）
        first_beat = int(round(query_beat_raw[0]))
        query_beat = [first_beat] + [int(round(b)) for b in query_beat_raw]
    elif len(onsets) == 2: # 恰好兩個 onset（一個音符）
        # 沒有間隔可計算 beat_raw，這裡設為一個預設值或空值
        # query_beat = [1] # 例如，假設最短的音符節拍為 1
        pass # 保持 query_beat 為空，讓後續比較失敗
    
    return query_diff, query_beat

def find_closest_song(query_dir, target_file, hmm_prune_gap=0.75):
    """
    Find the closest song for each query audio file by calculating edit distance.
    Calculate the accuracy based on correct matches.

    hmm_prune_gap: 每個轉移的 HMM log 機率比最佳歌曲低超過此值就不跑 DP
                   (0.75 在 10/15/20 秒資料上不損失準確率，約剪掉 22% 候選；float('inf') = 不剪枝)
    """
    targets = load_target_data(target_file)
    results = []
    correct_count = 0  # Counter for correct matches
    total_count = 0    # Counter for total files

    # 只讀一次所有目標歌曲的 HMM；沒有 CSV 的歌直接由簡譜建模，避免 hmm_score=0 → log 變 -inf 而被誤判為最佳
    transition_matrices = {}
    for filename in os.listdir(transition_matrix_folder):
        if filename.endswith(".csv"):
            song_name = filename[:-4]  # 移除 .csv 副檔名
            filepath = os.path.join(transition_matrix_folder, filename)
            transition_matrices[song_name] = np.loadtxt(filepath, delimiter=",", dtype=float)
    for song_name, target_data in targets.items():
        if song_name not in transition_matrices:
            _, transition_matrices[song_name], _ = HMM.build_markov_model(target_data[0], min_prob=0.001, state_range=(-11, 11))

    for audio_file in os.listdir(query_dir):
        if audio_file.endswith('.wav'):  # Adjust file extension as needed
            file_path = os.path.join(query_dir, audio_file) # Convert to full path

            print(f"Processing {audio_file}...")
            query_diff, query_beat = process_audio_to_query_diff(file_path)

            best_match = None
            best_score = float('inf')

            # HMM 剪枝：先算所有歌的 HMM log 分數，與最佳者的「每個轉移平均差距」> hmm_prune_gap 的歌跳過 DP
            # (log 分數隨 query 長度累加，所以除以轉移數做正規化)
            hmm_log_scores = {name: HMM.calculate_log_score(query_diff, transition_matrices[name]) for name in targets}
            best_hmm = max(hmm_log_scores.values())
            n_transitions = max(len(query_diff) - 1, 1)

            for song_name, target_data in targets.items():
                hmm_log_score = hmm_log_scores[song_name]
                if (best_hmm - hmm_log_score) / n_transitions > hmm_prune_gap:
                    continue

                target_diff = target_data[0]
                target_beat = target_data[1]
                distanceD, _ = DP.calculate_edit_distance(query_diff, target_diff, d=3)  # Cost 3 is the best
                distanceB, _ = DP.calculate_edit_distance(query_beat, target_beat, d=3)  # Cost 3 is the best

                score = 0.95 * distanceD + 0.0125 * distanceB + 0.0375 * hmm_log_score
                if score < best_score:
                    best_score = score
                    best_match = song_name

            # Check if the predicted best match is correct
            audio_name = os.path.splitext(audio_file)[0].split("_")[1]
            if audio_name == best_match:
                correct_count += 1

            results.append((audio_file, best_match, best_score))
            total_count += 1

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0

    print("--------------------------Now, Final Results in", query_dir, "is--------------------------")
    print(f"Accuracy: {accuracy:.2%}")
    return results


if __name__ == "__main__":
    # query_dir = r"C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/10"
    # query_dir = r"C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/15"
    query_dir = r"hummingdata/20"
    transition_matrix_folder = r"Project_in_Lin's_thesis/HMM_midi_diff"
    target_file = "hummingdata/Target_tempo_50_utf-8.txt"

    results = find_closest_song(query_dir, target_file)

    for audio_file, best_match, best_score in results:
        print(f"Audio File: {audio_file}, Best Match: {best_match}, Score: {best_score:.2f}")