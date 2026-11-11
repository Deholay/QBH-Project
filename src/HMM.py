import numpy as np
from scipy.signal import convolve2d

def apply_diffusion_by_conv(transition_counts, diffusion_matrix):
    """
    對 transition_counts 應用 diffusion kernel, 並考慮每個 entry 在自己 column 中的權重比例。
    
    參數:
        - transition_counts: numpy.ndarray, 原始轉移次數矩陣
        - diffusion_matrix: numpy.ndarray, 擴散矩陣 (必須為方陣)
    
    回傳:
        - numpy.ndarray, 經過擴散處理的轉移次數矩陣
    """
    column_sums = np.sum(transition_counts, axis=0, keepdims=True)  # 計算每列總和並保持維度
    column_sums[column_sums == 0] = 1  # 避免除以 0
    normalized_counts = transition_counts / column_sums  # 依 column normalize

    # 進行 2D 卷積，使用 'same' 以保持矩陣大小
    diffused_counts = convolve2d(normalized_counts, diffusion_matrix, mode='same', boundary='fill', fillvalue=0)

    return diffused_counts

def apply_diffusion_by_mult(transition_counts, diffusion_matrix):
    """
    對 transition_counts 應用 diffusion kernel，並考慮每個 entry 在自己 column 中的權重比例。
    
    參數:
        - transition_counts: numpy.ndarray, 原始轉移次數矩陣
        - diffusion_matrix: numpy.ndarray, 擴散矩陣 (必須為方陣)
    
    回傳:
        - numpy.ndarray, 經過擴散處理的轉移次數矩陣
    """
    n = transition_counts.shape[0]
    k = diffusion_matrix.shape[0]  # Kernel 大小
    k_half = k // 2  # Kernel 半徑

    column_sums = np.sum(transition_counts, axis=0)  # 計算每個 column 的總和
    new_counts = np.zeros_like(transition_counts, dtype=float)

    for j in range(n):
        if column_sums[j] == 0:  # 避免除以 0
            continue
        
        for i in range(n):
            if transition_counts[i, j] > 0:
                # 計算該 entry 在 column 內的比例
                normalized_weight = transition_counts[i, j] / column_sums[j]
                
                # 套用 diffusion matrix
                for di in range(-k_half, k_half + 1):
                    for dj in range(-k_half, k_half + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n:
                            new_counts[ni, nj] += normalized_weight * diffusion_matrix[di + k_half, dj + k_half] * column_sums[j]  # 重新放大

    return new_counts

def build_markov_model(target_diff, mode = "conv", min_prob=0.001, state_range=(-11, 11)):
    """
    建立 Markov Model 的轉移機率矩陣，應用 diffusion kernel, 並確保 state 範圍涵蓋指定區間, 
    考慮每個 entry 在自己 column 中的權重比例。
    
    參數:
        - target_diff: list[int]，目標歌曲的 MIDI difference
        - min_prob: float, 當轉移機率為 0 時的最小值
        - state_range: tuple(int, int), (最小狀態, 最大狀態)
    
    回傳:
        - states: list[int]，所有 MIDI 狀態 (從小到大排序)
        - transition_matrix: numpy.ndarray, 轉移機率矩陣
        - state_index: dict, MIDI Number -> Matrix Index 的映射
    """
    # 確保 state 範圍完整
    states = list(range(state_range[0], state_range[1] + 1))        # Range is -11 ~ 11
    state_index = {midi: i for i, midi in enumerate(states)}
    n = len(states)

    # 初始化轉移次數矩陣
    transition_counts = np.zeros((n, n))

    # 計算轉移次數 C(s' | s)
    for i in range(len(target_diff) - 1):
        s, s_next = target_diff[i], target_diff[i + 1]
        if s in state_index and s_next in state_index:
            transition_counts[state_index[s_next], state_index[s]] += 1

    # print(np.round(transition_counts, decimals=4)) # [Debug]

    # 定義 diffusion matrix
    diffusion_matrix = np.array([
        [0.0062, 0.0166, 0.0332, 0.0166, 0.0062],
        [0.0166, 0.0443, 0.0886, 0.0443, 0.0166],
        [0.0332, 0.0886, 0.1773, 0.0886, 0.0322],
        [0.0166, 0.0443, 0.0886, 0.0443, 0.0166],
        [0.0062, 0.0166, 0.0332, 0.0166, 0.0062]
    ])

    # 套用 diffusion kernel
    if mode == "conv":
        transition_matrix = apply_diffusion_by_conv(transition_counts, diffusion_matrix)
    elif mode == "mult":
        transition_matrix = apply_diffusion_by_mult(transition_counts, diffusion_matrix)
    
    # 正規化
    # transition_matrix /= transition_matrix.sum(axis=0, keepdims=True)
    column_sums = np.sum(transition_matrix, axis=0, keepdims=True)  # 計算每列總和並保持維度
    column_sums[column_sums == 0] = 1  # 避免除以 0
    transition_matrix = transition_matrix / column_sums  # 依 column normalize
    # 設置最小機率 P_m
    transition_matrix[transition_matrix == 0] = min_prob

    return states, transition_matrix, state_index

def get_transition_probability(transition_matrix, s, s_next, min_prob=0.001, state_range=(-11, 11)):
    """
    :param transition_matrix: 已建構的轉移機率矩陣
    :param s: 當前狀態 (MIDI diff)
    :param s_next: 下一個狀態 (MIDI diff)
    :param min_prob: 預設最小機率
    :return: P(s' | s)
    """
    if (s < state_range[0] or s > state_range[1]) or (s_next < state_range[0] or s_next > state_range[1]):
        return min_prob  # 若狀態不在模型中，則直接回傳 min_prob
    i, j = int(s + abs(state_range[0])), int(s_next + abs(state_range[0]))
    # build_markov_model 存的是 matrix[next, cur]（依 column = cur 正規化）
    return transition_matrix[j, i]

def calculate_log_score(query_diff, transition_matrix):
    """
    根據 HMM 計算 query 的 log 匹配分數 log(score_H)，以 log 累加避免長 query 連乘 underflow 成 0
    """
    log_score = 0.0
    for i in range(len(query_diff) - 1):
        s, s_next = query_diff[i], query_diff[i+1]
        log_score += np.log(get_transition_probability(transition_matrix, s, s_next))

    return log_score

def calculate_score(query_diff, transition_matrix):
    """
    根據 HMM 計算 query 的匹配分數 score_H
    """
    return np.exp(calculate_log_score(query_diff, transition_matrix))



# 測試範例
if __name__ == "__main__":

    # 設定 target MIDI 數據，建立 HMM
    target_diff = [0, 7, 0, 2, 0, -2, -2, 0, -1, 0, -2, 0, -2]
    states, transition_matrix, state_index = build_markov_model(target_diff, min_prob=0.001, state_range=(-11, 11))
    print("States:", states)
    print("State Index:", state_index)
    print(get_transition_probability(transition_matrix, 0, 2))
    print(np.round(transition_matrix, decimals=4))

    # 設定 query MIDI 數據
    query_diff = [0, 0, 2, 0, -2, -2, -1, -1, 0, -2, 0, -2]

    # 計算 score_H
    score = calculate_score(query_diff, transition_matrix)
    print("Score_H:", score)
