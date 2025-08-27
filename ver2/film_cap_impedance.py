import numpy as np
import matplotlib.pyplot as plt

def estimate_film_cap_parasitics(capacitance_uF):
    """
    필름 커패시터의 기생 성분 추정 (X2 박스 타입 기준)
    capacitance_uF: 커패시턴스 (μF 단위)
    """
    #C = capacitance_uF * 1e-6  # F
    C = capacitance_uF  # F

    if C <= 0.001:
        ESR = 1.5
        ESL = 7e-9
    elif C <= 0.01:
        ESR = 0.8
        ESL = 10e-9
    elif C <= 0.1:
        ESR = 0.3
        ESL = 15e-9
    elif C <= 0.47:
        ESR = 0.15
        ESL = 25e-9
    elif C <= 1.0:
        ESR = 0.08
        ESL = 30e-9
    elif C <= 10.0:
        ESR = 0.05
        ESL = 40e-9
    else:
        ESR = 0.04
        ESL = 50e-9

    f_srf = 1 / (2 * np.pi * np.sqrt(ESL * C)) if ESL * C > 0 else float('inf') # ESL*C가 0일 때 오류 방지
    return ESR, ESL, f_srf

def calculate_capacitor_impedance(freq, capacitance_uF):
    """
    커패시터의 주파수별 임피던스 계산 (1개 기준)
    freq: 주파수 배열 [Hz]
    capacitance_uF: 커패시턴스 [μF]
    return: 복소수 임피던스 Z(f)
    """
    ESR, ESL, _ = estimate_film_cap_parasitics(capacitance_uF)  # ESR, ESL 자동 계산

    C = capacitance_uF * 1e-6
    omega = 2 * np.pi * freq
    Z_C = 1 / (1j * omega * C)
    Z_L = 1j * omega * ESL
    Z_total = ESR + Z_L + Z_C
    return Z_total

# 새로 추가할 함수
def get_film_capacitor_details(capacitance_uF: float) -> dict:
    """
    필름 커패시터의 ESR, ESL, SRF 등의 상세 정보를 딕셔너리 형태로 반환합니다.
    """
    esr, esl, srf = estimate_film_cap_parasitics(capacitance_uF)
    return {
        "ESR": esr,
        "ESL": esl,
        "SRF": srf,
        # 필요한 경우 다른 상세 정보 추가
    }

if __name__ == "__main__":
    # 주파수 범위 설정
    freq = np.logspace(3, 8, 1000)  # 1kHz ~ 100MHz

    # 커패시턴스 지정 (예: 0.01 µF)
    cap_uF = 1

    # 결과 출력
    ESR, ESL, f_srf = estimate_film_cap_parasitics(cap_uF)
    print(f"\n필름 커패시터 용량: {cap_uF} μF")
    print(f"기생 ESR: {ESR:.3f} Ω")
    print(f"기생 ESL: {ESL*1e9:.1f} nH")
    print(f"SRF (자기공진주파수): {f_srf/1e6:.2f} MHz")

    # 임피던스 계산
    Z = calculate_capacitor_impedance(freq, cap_uF)

    # 시각화
    plt.figure(figsize=(8, 5))
    plt.loglog(freq, np.abs(Z), label="|Z|")
    plt.axvline(f_srf, color='r', linestyle='--', label="SRF")
    plt.title(f"Impedance of Film Capacitor ({cap_uF} μF)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Impedance Magnitude [Ω]")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()