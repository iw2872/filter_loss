#!/usr/bin/env python3
"""
plot_cm_dm_s21_log.py

- combined.s4p 파일에서 CM 모드 및 DM 모드 S21 추출 및 그래프
- x축: 로그 스케일 (주파수)
"""

import numpy as np
import matplotlib.pyplot as plt

s = 1.0 / np.sqrt(2.0)

# -----------------------------
# 1. s4p 파일 읽기
# -----------------------------
def read_s4p(filename):
    freqs = []
    S_list = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!') or line.startswith('#'):
                continue
            toks = line.split()
            freq = float(toks[0])
            nums = [float(x) for x in toks[1:]]
            S = np.zeros((4,4), dtype=complex)
            k = 0
            for r in range(4):
                for c in range(4):
                    S[r,c] = nums[k] + 1j*nums[k+1]
                    k += 2
            freqs.append(freq)
            S_list.append(S)
    return np.array(freqs), S_list

# -----------------------------
# 2. Mixed-mode 변환 행렬 정의
# -----------------------------
# Ordering: [c1, c2, d1, d2]
M = s * np.array([
    [1, 1, 0, 0],  # c1 = (a1 + a2)/√2
    [0, 0, 1, 1],  # c2 = (a3 + a4)/√2
    [1, -1, 0, 0], # d1 = (a1 - a2)/√2
    [0, 0, 1, -1], # d2 = (a3 - a4)/√2
], dtype=complex)
Minv = np.linalg.inv(M)

# -----------------------------
# 3. CM 모드 S21 계산
# -----------------------------
def extract_cm_s21(S4_list):
    cm_s21 = []
    for S in S4_list:
        S_mm = M @ S @ Minv
        cm_s21.append(S_mm[0,1])  # c1 → c2
    return np.array(cm_s21)

# -----------------------------
# 4. DM 모드 S21 계산
# -----------------------------
def extract_dm_s21(S4_list):
    dm_s21 = []
    for S in S4_list:
        S_mm = M @ S @ Minv
        dm_s21.append(S_mm[2,3])  # d1 → d2
    return np.array(dm_s21)

# -----------------------------
# 5. 실행
# -----------------------------
s4p_file = "combined.s4p"
freqs, S4_list = read_s4p(s4p_file)
cm_s21 = extract_cm_s21(S4_list)
dm_s21 = extract_dm_s21(S4_list)

cm_s21_db = 20*np.log10(np.abs(cm_s21))
dm_s21_db = 20*np.log10(np.abs(dm_s21))

# -----------------------------
# 6. 그래프 (로그 x축)
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(freqs, cm_s21_db, label="CM S21")
plt.plot(freqs, dm_s21_db, label="DM S21")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlabel("Frequency [Hz]")
plt.ylabel("|S21| [dB]")
plt.title("Common Mode and Differential Mode S21 from combined.s4p")
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.show()
