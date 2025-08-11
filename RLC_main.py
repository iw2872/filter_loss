import numpy as np
import matplotlib.pyplot as plt
from dm_inductor_impedance import calculate_dm_inductor_impedance
from cm_inductor_impedance import calculate_cm_inductor_impedance
import film_cap_impedance
import ceramic_cap_impedance  # 별도 구현 필요

# 주파수 범위 설정
freq = np.logspace(3, 8, 500)

# DM 인덕터 및 CM 인덕터 기본 설정
core_name_dm = "CH102125G"
N_dm = 30  # DM 인덕터 권선수

core_name_cm = "302015_SR7"
N_cm = 7  # CM 인덕터 권선수

# 필름, 세라믹 커패시터 용량을 별도 입력 받음
cap_info_film = {
    "type": "film",
    "value_uF": 1
}

cap_info_ceramic = {
    "type": "ceramic",
    "value_uF": 0.1
}

try:
    # DM 인덕터 계산 (반환값 수정)
    result_dm = calculate_dm_inductor_impedance(core_name_dm, freq, N_dm)

    # 반환값이 딕셔너리라면, 키를 사용하여 값을 접근
    Z_dm = result_dm.get('Z_DM', None)
    f_srf_dm = result_dm.get('SRF', None)
    Z_cm_from_dm = result_dm.get('Z_CM_total', None)  # 'Z_CM'에서 'Z_CM_total'로 수정
    R_core = result_dm.get('R_core', None)
    R_cu = result_dm.get('R_cu', None)
    L = result_dm.get('L', None)
    L_ESL = result_dm.get('L_ESL', None)
    C_parasitic = result_dm.get('C_parasitic', None)

    print(f"\nDM 코어 '{core_name_dm}' SRF: {f_srf_dm / 1e6:.2f} MHz, Turns: {N_dm}")

    # L이 None이 아닌 경우에만 계산하도록 수정
    if L is not None:
        print(f"DM 코어 인덕턴스 (DC): {L * 1e6:.2f} μH")
    else:
        print("DM 코어 인덕턴스 (DC)가 계산되지 않았습니다.")

    # CM 인덕터 계산
    cm_results = calculate_cm_inductor_impedance(core_name_cm, N_cm, freq)
    freq_cm = cm_results['freq']
    Z_cm = cm_results['Z_CM_total']
    Z_dm_in_cm = cm_results['Z_DM']
    f_srf_cm = cm_results['f_srf']
    print(f"\nCM 코어 '{core_name_cm}' SRF: {f_srf_cm / 1e6:.2f} MHz, Turns: {N_cm}")

    print("[CM, DM 인덕턴스 (μH)]")
    for f, L_CM, L_DM in cm_results['inductance_at_freqs']:
        print(f"{f / 1e3:.0f} kHz: CM = {L_CM:.2f} μH, DM = {L_DM:.2f} μH")

    # 필름 커패시터 임피던스 계산
    ESR_film, ESL_film, f_srf_film = film_cap_impedance.estimate_film_cap_parasitics(cap_info_film["value_uF"])
    Z_film = film_cap_impedance.calculate_capacitor_impedance(freq, cap_info_film["value_uF"])
    print(
        f"\nFilm 커패시터: {cap_info_film['value_uF']} μF \nSRF: {f_srf_film / 1e6:.2f} MHz, ESR: {ESR_film:.3f} Ω, ESL: {ESL_film * 1e9:.1f} nH")

    # 세라믹 커패시터 임피던스 계산
    ESR_ceramic, ESL_ceramic, f_srf_ceramic = ceramic_cap_impedance.estimate_ceramic_cap_parasitics(cap_info_ceramic["value_uF"])
    Z_ceramic = ceramic_cap_impedance.calculate_capacitor_impedance(freq, cap_info_ceramic["value_uF"])
    print(f"\nCeramic 커패시터: {cap_info_ceramic['value_uF']} μF \nSRF: {f_srf_ceramic / 1e6:.2f} MHz, ESR: {ESR_ceramic:.3f} Ω, ESL: {ESL_ceramic * 1e9:.1f} nH")

    # 결과 그래프 출력
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs = axs.flatten()

    # DM 인덕터 임피던스
    axs[0].loglog(freq, np.abs(Z_dm), label="DM Inductor DM Impedance")

    # Z_cm_from_dm이 None이 아닌 경우에만 loglog 호출
    if Z_cm_from_dm is not None:
        axs[0].loglog(freq, np.abs(Z_cm_from_dm), label="DM Inductor CM Mode Impedance", linestyle='--')
    else:
        print("Z_cm_from_dm is None, skipping loglog for CM Mode Impedance")

    axs[0].axvline(f_srf_dm, color='r', linestyle='--', label=f"DM SRF {f_srf_dm / 1e6:.2f} MHz")
    axs[0].set_title(f"DM Inductor Impedance - {core_name_dm}")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Impedance (Ω)")
    axs[0].grid(True, which='both', linestyle='--')
    axs[0].legend()

    # CM 인덕터 임피던스
    axs[1].loglog(freq_cm, np.abs(Z_cm), label="CM Inductor CM Impedance")
    axs[1].loglog(freq_cm, np.abs(Z_dm_in_cm), label="CM Inductor DM Impedance", linestyle='--')
    axs[1].axvline(f_srf_cm, color='m', linestyle=':', label=f"CM SRF {f_srf_cm / 1e6:.2f} MHz")
    axs[1].set_title(f"CM Inductor Impedance - ANB {core_name_cm}")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Impedance (Ω)")
    axs[1].grid(True, which='both', linestyle='--')
    axs[1].legend()

    # 필름 커패시터 임피던스
    axs[2].loglog(freq, np.abs(Z_film), label=f"Film Capacitor ({cap_info_film['value_uF']} μF) Impedance")
    axs[2].axvline(f_srf_film, color='orange', linestyle='--', label=f"SRF {f_srf_film / 1e6:.2f} MHz")
    axs[2].set_title(f"Film Capacitor Impedance - {cap_info_film['value_uF']} uF")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("Impedance (Ω)")
    axs[2].grid(True, which='both')
    axs[2].legend()

    # 세라믹 커패시터 임피던스
    axs[3].loglog(freq, np.abs(Z_ceramic), label=f"Ceramic Capacitor ({cap_info_ceramic['value_uF']} μF) Impedance")
    axs[3].axvline(f_srf_ceramic, color='green', linestyle='--', label=f"SRF {f_srf_ceramic / 1e6:.2f} MHz")
    axs[3].set_title(f"Ceramic Capacitor Impedance - {cap_info_ceramic['value_uF']} uF")
    axs[3].set_xlabel("Frequency (Hz)")
    axs[3].set_ylabel("Impedance (Ω)")
    axs[3].grid(True, which='both')
    axs[3].legend()

    plt.tight_layout()
    plt.show()

except Exception as e:
    import traceback

    print("Error occurred:", e)
    traceback.print_exc()
