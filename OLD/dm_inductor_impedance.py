# dm_inductor_impedance.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


def calculate_dm_inductor_impedance(core_name, freq, N,
                                    wire_diameter_mm=1.0,
                                    insulation_thickness_mm=0.5,
                                    I_rms=1.0, leakage_ratio=0.02):
    # 코어 데이터 로드
    try:
        core_df = pd.read_csv("dm_CH_core_db.csv")
        core = core_df[core_df['core_name'] == core_name].iloc[0]
    except Exception as e:
        print(f"코어 '{core_name}'에 대한 데이터를 찾을 수 없습니다. 오류: {e}")
        return None  # 코어 데이터가 없으면 None 반환

    # 코어 파라미터
    OD = float(core["OD"]) / 10  # mm에서 cm로 변환
    ID = float(core["ID"]) / 10
    HT = float(core["HT"]) / 10
    AL = float(core["AL"]) * 1e-9  # nH에서 H로 변환
    core_loss_density = float(core["core_loss_density"])

    # 상수
    epsilon_0 = 8.854e-14  # F/cm
    epsilon_r = 3.5  # 상대 유전율

    # 사용자 입력값 (cm로 변환)
    wire_diameter = wire_diameter_mm / 10
    insulation_thickness = insulation_thickness_mm / 10

    # 계산
    V_core = np.pi * ((OD ** 2 - ID ** 2) / 4) * HT
    P_core = core_loss_density * V_core
    R_core = (P_core / 1000) / (I_rms ** 2)  # Ω
    L_dc = AL * N ** 2  # ✨ 추가/수정된 부분: L_dc 명확히 정의

    mean_length = np.pi * (OD + ID) / 2
    wire_length = N * mean_length

    rho_cu = 1.68e-6  # Ohm·cm
    wire_radius = wire_diameter / 2
    A_cu = np.pi * wire_radius ** 2
    R_cu = (rho_cu * wire_length) / A_cu

    # 기생 성분
    L_ESL = wire_length * 1e-9  # H (단위 길이당 인덕턴스에 기반한 추정치)

    area = wire_length * wire_diameter
    C_parasitic = epsilon_0 * epsilon_r * area / (insulation_thickness + 1e-9)  # 0으로 나누는 것 방지

    # SRF 계산을 위한 DM 인덕터 총 인덕턴스 (L_dc + L_ESL)
    L_total_dm_srf = L_dc + L_ESL
    f_srf_dm = 1 / (2 * np.pi * np.sqrt(L_total_dm_srf * C_parasitic))  # Hz

    # DM 임피던스 계산
    omega = 2 * np.pi * freq
    Z_series_dm = (R_core + R_cu) + 1j * omega * L_dc
    Z_c = 1 / (1j * omega * C_parasitic)
    Z_total_dm = 1 / (1 / Z_series_dm + 1 / Z_c + 1e-18)  # 0으로 나누는 것 방지

    # CM 임피던스 계산 (DM 인덕터의 CM 특성)
    L_cm_from_dm_leakage = L_dc * leakage_ratio  # DM 인덕터의 CM 누설 인덕턴스
    L_total_cm_from_dm = L_cm_from_dm_leakage + L_ESL  # DM 인덕터의 CM 총 인덕턴스
    Z_series_cm_from_dm = (R_core + R_cu) + 1j * omega * L_total_cm_from_dm
    Z_total_cm_from_dm = 1 / (1 / Z_series_cm_from_dm + 1 / Z_c + 1e-18)  # 0으로 나누는 것 방지

    # Z_dm 및 Z_cm_from_dm을 복소수 배열로 변환
    Z_dm = np.array(Z_total_dm, dtype=complex)
    Z_cm_from_dm = np.array(Z_total_cm_from_dm, dtype=complex)

    # 1kHz, 10kHz, 100kHz에서의 CM 및 DM 인덕턴스 값 계산
    specific_freqs_khz = [1, 10, 100]
    inductance_at_freqs = []

    for f_khz in specific_freqs_khz:
        f_hz = f_khz * 1000
        # 이 부분은 L_dc를 활용하므로, 사실상 이 주파수에서 L_dc와 L_cm_from_dm_leakage 값을 그대로 사용합니다.
        # 실제 임피던스에서 역산하여 유효 인덕턴스를 얻으려면 더 복잡한 계산이 필요합니다.
        # 여기서는 저주파 영역에서 L_dc와 L_cm_from_dm_leakage가 인덕턴스를 지배한다고 가정합니다.
        inductance_at_freqs.append((f_hz, L_total_cm_from_dm * 1e6, L_dc * 1e6))  # (주파수, CM 인덕턴스(uH), DM 인덕턴스(uH))

    return {
        "freq": freq,
        "Z_CM_total": Z_cm_from_dm,
        "Z_DM": Z_dm,
        "f_srf": f_srf_dm,
        "R_cu": R_cu,
        "L_ESL": L_ESL,
        "C_parasitic": C_parasitic,
        "inductance_at_freqs": inductance_at_freqs,
        "core_name": core["core_name"],
        "N": N,
        "ESR": R_cu,
        "ESL": L_ESL,
        "SRF": f_srf_dm,
        "L_dc": L_dc,  # ✨ 추가된 부분: L_dc 명시적으로 반환
        "L_cm_from_dm_leakage": L_cm_from_dm_leakage  # ✨ 추가된 부분: DM 인덕터의 CM 누설 인덕턴스 반환
    }


if __name__ == "__main__":
    # Configure plotting style
    plt.rcParams['axes.unicode_minus'] = False
    rc('mathtext', fontset='cm')

    # Frequency range for analysis (log scale from 1kHz to 100MHz)
    freq_points = 500
    freq_min = 1e3  # 1kHz
    freq_max = 100e6  # 100MHz
    freq = np.logspace(np.log10(freq_min), np.log10(freq_max), freq_points)

    core_name = "CH102125G"  # Core model name
    N = 30  # Turns

    try:
        # Calculate impedance values
        results = calculate_dm_inductor_impedance(core_name, freq, N)

        if results:
            # Print results
            print(f"\n--- DM 인덕터 '{results['core_name']}' (N={results['N']}) 계산 결과 ---")
            print(f"DM 인덕터 자기공진주파수 (SRF): {results['f_srf'] / 1e6:.2f} MHz")
            print(f"권선 저항 (R_cu): {results['R_cu']:.6f} Ω")
            print(f"**DC 인덕턴스 (L_dc): {results['L_dc'] * 1e6:.3f} μH**")  # ✨ 수정된 출력
            print(f"기생 ESL: {results['L_ESL'] * 1e9:.3f} nH")
            print(f"기생 커패시턴스: {results['C_parasitic'] * 1e12:.3f} pF")
            print(f"**CM 누설 인덕턴스 (L_cm_from_dm_leakage): {results['L_cm_from_dm_leakage'] * 1e6:.3f} μH**")  # ✨ 추가된 출력

            print("\n--- 특정 주파수에서의 인덕턴스 값 (참고용) ---")
            for f_hz, L_CM_uH, L_DM_uH in results['inductance_at_freqs']:
                print(f"{f_hz / 1e3:.0f} kHz: CM 인덕턴스 = {L_CM_uH:.2f} μH, DM 인덕턴스 = {L_DM_uH:.2f} μH")

            # Plot impedance vs frequency for DM and CM modes
            plt.figure(figsize=(8, 5))
            plt.loglog(results['freq'], np.abs(results['Z_DM']), label="DM Impedance")
            plt.loglog(results['freq'], np.abs(results['Z_CM_total']), label="CM Impedance (from DM Inductor)",
                       linestyle='--')
            plt.axvline(results['f_srf'], color='r', linestyle=':', label=f"DM SRF ~ {results['f_srf'] / 1e6:.2f} MHz")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Impedance Magnitude (Ω)")
            plt.title(f"DM/CM Impedance ({results['core_name']}), Turns N={results['N']}")
            plt.legend()
            plt.grid(True, which='both', linestyle='--')
            plt.tight_layout()
            plt.show()

        else:
            print("인덕터 임피던스 계산에 실패했습니다. (결과가 None)")

    except Exception as e:
        print("오류 발생:", e)