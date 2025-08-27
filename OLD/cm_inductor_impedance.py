# cm_inductor_impedance.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

def calculate_cm_inductor_impedance(model_name, N, freq,
                                    leakage_ratio=0.02,
                                    wire_diameter_mm=1.0,
                                    insulation_thickness_mm=0.5,
                                    csv_filename='cm_ANB_core_db.csv'):
    # CSV에서 코어 데이터 불러오기
    df_cores = pd.read_csv(csv_filename)
    if model_name not in df_cores['core_name'].values:
        raise ValueError(f"모델명 '{model_name}'이 CSV에 없습니다.")

    core_row = df_cores[df_cores['core_name'] == model_name].iloc[0]

    core_data = {
        "core_name": core_row['core_name'],
        "OD": core_row['OD'],
        "ID": core_row['ID'],
        "HT": core_row['HT'],
        "AL_values": [core_row['AL1kHz'], core_row['AL10kHz'], core_row['AL100kHz']],
        "frequencies": [1e3, 10e3, 100e3]
    }

    omega = 2 * np.pi * freq

    # mm → cm 변환
    OD_cm = core_data["OD"] / 10
    ID_cm = core_data["ID"] / 10
    HT_cm = core_data["HT"] / 10

    # AL 값 로그 스케일 2차 보간 함수
    log_freqs = np.log10(core_data["frequencies"])
    log_AL_values = np.log10(core_data["AL_values"])
    interp_func = interp1d(log_freqs, log_AL_values, kind='quadratic', fill_value="extrapolate")

    def interpolate_AL(f):
        log_f = np.log10(f)
        log_AL = interp_func(log_f)
        return 10**log_AL * 1e-6  # μH → H

    def calc_inductance(N, f):
        AL_interp = interpolate_AL(f)  # H
        return AL_interp * N**2

    L_array = calc_inductance(N, freq)  # CM 인덕턴스 (H)

    # 권선 길이 및 권선 저항
    mean_length = np.pi * (OD_cm + ID_cm) / 2
    wire_length = N * mean_length

    wire_diameter_cm = wire_diameter_mm / 10
    wire_radius = wire_diameter_cm / 2
    A_cu = np.pi * wire_radius**2
    rho_cu = 1.68e-6  # Ω·cm
    R_cu = (rho_cu * wire_length) / A_cu

    # 기생 ESL, 커패시턴스
    L_per_cm = 1e-9
    L_ESL = wire_length * L_per_cm  # H

    epsilon_0 = 8.854e-14
    epsilon_r = 1
    insulation_thickness_cm = insulation_thickness_mm / 10

    area = wire_length * wire_diameter_cm
    C_parasitic = epsilon_0 * epsilon_r * area / insulation_thickness_cm

    # CM 임피던스
    Z_series_CM = R_cu + 1j * omega * (L_array + L_ESL)
    Z_c = 1 / (1j * omega * C_parasitic)
    Z_CM_total = 1 / (1 / Z_series_CM + 1 / Z_c)

    # DM 임피던스 (누설 인덕턴스 포함)
    L_leakage = leakage_ratio * L_array
    Z_DM = R_cu + 1j * omega * (L_leakage + L_ESL)

    # SRF 계산 (1 kHz AL 기준)
    L_1kHz = core_data["AL_values"][0] * 1e-6 * N**2
    L_total = L_1kHz + L_ESL
    f_srf = 1 / (2 * np.pi * np.sqrt(L_total * C_parasitic))

    # 특정 주파수에서의 인덕턴스 (μH 단위)
    f_test = [1e3, 10e3, 100e3]
    inductance_at_freqs = []
    for f in f_test:
        L_CM = interpolate_AL(f) * N**2 * 1e6  # μH
        L_DM = leakage_ratio * interpolate_AL(f) * N**2 * 1e6  # μH
        inductance_at_freqs.append((f, L_CM, L_DM))

    return {
        "freq": freq,
        "Z_CM_total": Z_CM_total,
        "Z_DM": Z_DM,
        "f_srf": f_srf,
        "R_cu": R_cu,

        "L_ESL": L_ESL,
        "C_parasitic": C_parasitic,
        "inductance_at_freqs": inductance_at_freqs,
        "core_name": core_data['core_name'],
        "N": N,
        "ESR": R_cu,  # ESR을 권선 저항으로 반환
        "ESL": L_ESL,  # ESL을 기생 ESL로 반환
        "SRF": f_srf   # SRF을 계산한 값으로 반환
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    freq = np.logspace(3, 8, 500)
    model_name = "986545_SR7"
    N = 7

    try:
        results = calculate_cm_inductor_impedance(model_name, N, freq)

        print(f"\n권선수 N = {results['N']}")
        print(f"권선 저항 R_cu = {results['R_cu']:.4f} Ω")
        print(f"기생 ESL L_ESL = {results['L_ESL']*1e9:.3f} nH")
        print(f"기생 커패시턴스 C_parasitic = {results['C_parasitic']*1e12:.3f} pF")
        print(f"예상 자기공진주파수 SRF = {results['f_srf']/1e6:.3f} MHz\n")

        for f, L_CM, L_DM in results['inductance_at_freqs']:
            print(f"[{f/1e3:.0f} kHz] CM 인덕턴스 = {L_CM:.2f} μH,  DM 인덕턴스 = {L_DM:.2f} μH")

        plt.figure(figsize=(10, 6))
        plt.loglog(results["freq"], np.abs(results["Z_CM_total"]), label="|Z_CM_total|")
        plt.loglog(results["freq"], np.abs(results["Z_DM"]), label="|Z_DM|", linestyle='--')
        plt.axvline(results["f_srf"], color='red', linestyle=':', label=f"CM SRF ≈ {results['f_srf']/1e6:.2f} MHz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Impedance (Ω)")
        plt.title(f"CM vs DM Impedance ({results['core_name']}) with N={results['N']}")
        plt.grid(True, which='both', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("오류 발생:", e)
