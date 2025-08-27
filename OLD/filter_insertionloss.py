# filter_insertionloss.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from dataclasses import dataclass

# 외부 임피던스 계산 모듈 (이 파일들은 제공된 상태라고 가정합니다)
from dm_inductor_impedance import calculate_dm_inductor_impedance
from cm_inductor_impedance import calculate_cm_inductor_impedance
import film_cap_impedance
import ceramic_cap_impedance


# --- 부품 정의를 위한 데이터 클래스 ---
@dataclass
class FilterComponent:
    """모든 필터 구성 요소의 기본 클래스."""
    type: str


@dataclass
class Inductor(FilterComponent):
    """인덕터 유형 부품 정의."""
    core: str
    turns: int


@dataclass
class Capacitor(FilterComponent):
    """커패시터 유형 부품 정의."""
    cap_type: str  # "FILM" 또는 "CERAMIC"
    value_uF: float


# --- ABCD 행렬 변환 및 연산 함수 ---
def z_to_abcd(Z_array: np.ndarray, connection_type: str):
    """
    임피던스 배열 Z_array로부터 ABCD 행렬 배열을 생성합니다.
    Z_array는 각 주파수에서의 임피던스 값을 포함합니다.
    ABCD 행렬은 (num_frequencies, 2, 2) 형태의 Numpy 배열로 반환됩니다.
    """
    num_frequencies = Z_array.shape[0]
    abcd_matrices = np.zeros((num_frequencies, 2, 2), dtype=complex)

    if connection_type == "series":
        abcd_matrices[:, 0, 0] = 1  # A
        abcd_matrices[:, 0, 1] = Z_array  # B
        abcd_matrices[:, 1, 0] = 0  # C
        abcd_matrices[:, 1, 1] = 1  # D
    elif connection_type == "shunt":
        Y_array = 1 / (Z_array + 1e-18)  # 0으로 나누는 것을 방지
        abcd_matrices[:, 0, 0] = 1  # A
        abcd_matrices[:, 0, 1] = 0  # B
        abcd_matrices[:, 1, 0] = Y_array  # C
        abcd_matrices[:, 1, 1] = 1  # D
    else:
        raise ValueError("connection_type must be 'series' or 'shunt'")
    return abcd_matrices


def cascade_abcd_blocks(abcd_blocks: List[np.ndarray]):
    """
    여러 ABCD 행렬 배열을 연쇄적으로 곱합니다.
    abcd_blocks: 각 ABCD 행렬 배열의 리스트 (각 배열은 (num_frequencies, 2, 2) 형태)
    반환: (num_frequencies, 2, 2) 형태의 최종 ABCD 행렬 배열
    """
    if not abcd_blocks:
        return np.array([[[1, 0], [0, 1]]])

    total_abcd = abcd_blocks[0]

    for i in range(1, len(abcd_blocks)):
        current_block = abcd_blocks[i]
        total_abcd = total_abcd @ current_block

    return total_abcd


def abcd_to_s_parameters(abcd_matrices: np.ndarray, Zin: float = 50.0, Zout: float = 50.0):
    """
    ABCD 행렬 배열을 S-파라미터 배열 (S11, S12, S21, S22)로 변환합니다.
    abcd_matrices: (num_frequencies, 2, 2) 형태의 ABCD 행렬 배열
    반환: S11, S12, S21, S22 각각 (num_frequencies,) 형태의 배열
    """
    A = abcd_matrices[:, 0, 0]
    B = abcd_matrices[:, 0, 1]
    C = abcd_matrices[:, 1, 0]
    D = abcd_matrices[:, 1, 1]

    delta = A * Zout + B + C * Zin * Zout + D * Zin
    delta_safe = delta + 1e-18

    s11 = (A * Zout + B - C * Zin * Zout - D * Zin) / delta_safe
    s12 = (2 * (A * D - B * C) * np.sqrt(Zin * Zout)) / delta_safe
    s21 = (2 * np.sqrt(Zin * Zout)) / delta_safe
    s22 = (A * Zin + B - C * Zin * Zout - D * Zout) / delta_safe

    return s11, s12, s21, s22


def get_component_impedance(freq: np.ndarray, component: FilterComponent, mode: str):
    """
    주어진 부품의 주파수별 CM 및 DM 임피던스와 부품 상세 정보를 반환합니다.
    반환: (Z_cm, Z_dm, component_specific_info) 튜플
    - component_specific_info: 인덕터 또는 커패시터인 경우 필요한 상세 정보 딕셔너리
                               (예: 'L_dc', 'SRF', 'ESR', 'ESL' 등 포함)
                               그 외의 경우 None
    """
    component_specific_info = None

    # DM 인덕터 처리
    if isinstance(component, Inductor) and component.type == "DM_IND":
        result = calculate_dm_inductor_impedance(component.core, freq, component.turns)

        if result is None:
            print(f"Error: calculate_dm_inductor_impedance returned None for core {component.core}.")
            return np.full_like(freq, 1e9 + 0j, dtype=complex), np.full_like(freq, 1e9 + 0j, dtype=complex), None

        Z_DM_IND_cm = result.get("Z_CM_total")
        Z_DM_IND_dm = result.get("Z_DM")

        # DM 인덕터의 모든 유용한 정보를 딕셔너리로 저장
        component_specific_info = {
            "component_type": component.type,
            "core_name": component.core,
            "turns": component.turns,
            "L_dc": result.get("L_dc"),  # H
            "L_cm_from_dm_leakage": result.get("L_cm_from_dm_leakage"),  # H
            "SRF": result.get("f_srf"),  # Hz
            "ESL": result.get("L_ESL"),  # H
            "ESR": result.get("ESR"),  # Ohm
            "C_parasitic": result.get("C_parasitic"),  # F
        }

        if Z_DM_IND_dm is None:
            print(f"Warning: DM inductor impedance (DM mode) for core {component.core} is None. Using 1e9.")
            Z_DM_IND_dm = np.full_like(freq, 1e9 + 0j, dtype=complex)
        if Z_DM_IND_cm is None:
            print(f"Warning: DM inductor impedance (CM mode) for core {component.core} is None. Using 1e9.")
            Z_DM_IND_cm = np.full_like(freq, 1e9 + 0j, dtype=complex)

        return np.array(Z_DM_IND_cm, dtype=complex), np.array(Z_DM_IND_dm, dtype=complex), component_specific_info

    # CM 인덕터 처리
    if isinstance(component, Inductor) and component.type == "CM_IND":
        result = calculate_cm_inductor_impedance(component.core, component.turns, freq)

        if result is None:
            print(f"Error: calculate_cm_inductor_impedance returned None for core {component.core}.")
            return np.full_like(freq, 1e9 + 0j, dtype=complex), np.full_like(freq, 1e9 + 0j, dtype=complex), None

        Z_CM_IND_cm = np.array(result["Z_CM_total"], dtype=complex)
        Z_CM_IND_dm = np.array(result["Z_DM"], dtype=complex)

        # CM 인덕터의 모든 유용한 정보를 딕셔너리로 저장
        component_specific_info = {
            "component_type": component.type,
            "core_name": component.core,
            "turns": component.turns,
            "inductance_at_freqs": result.get("inductance_at_freqs"),  # List[(freq, L_CM_uH, L_DM_uH)]
            "SRF": result.get("SRF"),  # Hz
            "ESL": result.get("ESL"),  # H
            "ESR": result.get("ESR"),  # Ohm
            "C_parasitic": result.get("C_parasitic"),  # F
        }
        return Z_CM_IND_cm, Z_CM_IND_dm, component_specific_info

    # XCAP 처리
    if isinstance(component, Capacitor) and component.type == "XCAP":
        if component.cap_type == "FILM":
            Z_XCAP_dm = np.array(film_cap_impedance.calculate_capacitor_impedance(freq, component.value_uF),
                                 dtype=complex)
            # 필름 커패시터 상세 정보 가져오기
            component_specific_info = film_cap_impedance.get_film_capacitor_details(component.value_uF)
        elif component.cap_type == "CERAMIC":
            Z_XCAP_dm = np.array(ceramic_cap_impedance.calculate_capacitor_impedance(freq, component.value_uF),
                                 dtype=complex)
            # 세라믹 커패시터 상세 정보 가져오기
            component_specific_info = ceramic_cap_impedance.get_ceramic_capacitor_details(component.value_uF)
        else:
            raise ValueError(f"Unsupported capacitor type: {component.cap_type} for XCAP")

        # 커패시터 공통 정보 추가 (필수 필드)
        if component_specific_info is not None:
            component_specific_info.update({
                "component_type": component.type,
                "cap_type": component.cap_type,
                "value_uF": component.value_uF  # uF
            })

        Z_XCAP_cm = np.full_like(freq, 1e9 + 0j, dtype=complex)  # XCAP은 CM 모드에서 개방
        return Z_XCAP_cm, Z_XCAP_dm, component_specific_info

    # YCAP 처리
    if isinstance(component, Capacitor) and component.type == "YCAP":
        if component.cap_type == "FILM":
            Z_YCAP_cm = np.array(film_cap_impedance.calculate_capacitor_impedance(freq, component.value_uF),
                                 dtype=complex)
            # 필름 커패시터 상세 정보 가져오기
            component_specific_info = film_cap_impedance.get_film_capacitor_details(component.value_uF)
        elif component.cap_type == "CERAMIC":
            Z_YCAP_cm = np.array(ceramic_cap_impedance.calculate_capacitor_impedance(freq, component.value_uF),
                                 dtype=complex)
            # 세라믹 커패시터 상세 정보 가져오기
            component_specific_info = ceramic_cap_impedance.get_ceramic_capacitor_details(component.value_uF)
        else:
            raise ValueError(f"Unsupported capacitor type: {component.cap_type} for YCAP")

        # 커패시터 공통 정보 추가 (필수 필드)
        if component_specific_info is not None:
            component_specific_info.update({
                "component_type": component.type,
                "cap_type": component.cap_type,
                "value_uF": component.value_uF  # uF
            })

        Z_YCAP_dm = np.full_like(freq, 1e9 + 0j, dtype=complex)  # YCAP은 DM 모드에서 개방
        return Z_YCAP_cm, Z_YCAP_dm, component_specific_info

    print(f"Warning: Unknown component type {component.type}. Returning disabled impedance.")
    return np.full_like(freq, 1e9 + 0j, dtype=complex), np.full_like(freq, 1e9 + 0j, dtype=complex), None


def calculate_insertion_loss(
        freq: np.ndarray,
        topology: List[FilterComponent],
        mode: str,
        Zin: float = 50.0,
        Zout: float = 50.0
):
    """
    주어진 필터 토폴로지에 대해 CM 또는 DM 모드의 삽입 손실을 계산합니다.
    반환값:
    - insertion_loss_db (np.ndarray): 삽입 손실 값 (dB)
    - s11_values (np.ndarray): S11 복소수 값
    - s12_values (np.ndarray): S12 복소수 값
    - s21_values (np.ndarray): S21 복소수 값
    - s22_values (np.ndarray): S22 복소수 값
    - all_component_details (List[Dict]): 각 인덕터 및 커패시터의 계산된/정의된 상세 정보 리스트
    """

    component_abcd_blocks = []
    all_component_details = []  # 모든 부품의 상세 정보를 저장할 리스트

    for component in topology:
        Z_cm, Z_dm, comp_info = get_component_impedance(freq, component, mode)

        # 인덕터 또는 커패시터인 경우에만 정보를 수집
        if comp_info is not None:
            all_component_details.append(comp_info)

        Z_current_mode = Z_cm if mode == "CM" else Z_dm
        connection_type = "shunt" if component.type in ["XCAP", "YCAP"] else "series"

        component_abcd_blocks.append(z_to_abcd(Z_current_mode, connection_type))

    total_abcd_matrices = cascade_abcd_blocks(component_abcd_blocks)

    s11_values, s12_values, s21_values, s22_values = abcd_to_s_parameters(total_abcd_matrices, Zin, Zout)

    insertion_loss_db = -20 * np.log10(np.abs(s21_values) + 1e-18)

    # 모든 부품 정보 리스트를 반환
    return insertion_loss_db, s11_values, s12_values, s21_values, s22_values, all_component_details


# --- 로컬 테스트를 위한 if __name__ == "__main__": 블록 (수정된 반환 값 테스트 추가) ---
# --- 로컬 테스트를 위한 if __name__ == "__main__": 블록 (수정된 반환 값 테스트 추가) ---
if __name__ == "__main__":
    print("Running local test for filter_insertionloss.py functions...")

    Zin = 50
    Zout = 50 # <-- 여기는 이미 Zout으로 잘 정의되어 있습니다.
    start_freq = 10e3  # 10 kHz
    stop_freq = 100e6  # 100 MHz
    freq = np.logspace(np.log10(start_freq), np.log10(stop_freq), 1000)

    filter_topology = [
        Inductor(type="CM_IND", core="986545_SF7", turns=7),
        Capacitor(type="YCAP", cap_type="CERAMIC", value_uF=0.01),
        Inductor(type="DM_IND", core="CH102125G", turns=30),
        Capacitor(type="XCAP", cap_type="FILM", value_uF=0.1),
        Capacitor(type="YCAP", cap_type="FILM", value_uF=0.0022),
        Capacitor(type="XCAP", cap_type="CERAMIC", value_uF=2.2)
    ]

    try:
        # calculate_insertion_loss의 새로운 반환값 `all_component_details`를 받도록 수정
        # Zout=zout -> Zout=Zout 으로 수정
        IL_cm_db, s11_cm, s12_cm, s21_cm, s22_cm, all_component_details_cm = calculate_insertion_loss(freq, filter_topology,
                                                                                              mode="CM", Zin=Zin,
                                                                                              Zout=Zout) # <-- 여기 수정
        IL_dm_db, s11_dm, s12_dm, s21_dm, s22_dm, all_component_details_dm = calculate_insertion_loss(freq, filter_topology,
                                                                                              mode="DM", Zin=Zin,
                                                                                              Zout=Zout) # <-- 여기 수정

        fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        axs[0].semilogx(freq, IL_cm_db, label="CM Insertion Loss", color='blue')
        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Insertion Loss (dB)")
        axs[0].set_title("EMI Filter Insertion Loss - CM Mode")
        axs[0].grid(True, which='both', linestyle='--')
        axs[0].legend()
        axs[0].invert_yaxis()

        axs[1].semilogx(freq, IL_dm_db, label="DM Insertion Loss", color='red')
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_title("EMI Filter Insertion Loss - DM Mode")
        axs[1].grid(True, which='both', linestyle='--')
        axs[1].legend()
        axs[1].invert_yaxis()

        plt.tight_layout()
        plt.show()

        print("\nLocal test completed successfully. Check the plot window.")

        print(f"CM Mode S21 at lowest freq ({freq[0]:.2e} Hz): {s21_cm[0]:.4e}")
        print(f"DM Mode S21 at lowest freq ({freq[0]:.2e} Hz): {s21_dm[0]:.4e}")
        print(f"CM Mode S21 at highest freq ({freq[-1]:.2e} Hz): {s21_cm[-1]:.4e}")
        print(f"DM Mode S21 at highest freq ({freq[-1]:.2e} Hz): {s21_dm[-1]:.4e}")

        print("\n--- All Component Details (Common Mode Calculation) ---")
        if all_component_details_cm:
            for i, comp_info in enumerate(all_component_details_cm):
                comp_type = comp_info.get('component_type', 'N/A')
                print(f"Component {i + 1} (Type: {comp_type}):")

                if comp_type in ["CM_IND", "DM_IND"]:
                    core_name = comp_info.get('core_name', 'N/A')
                    turns = comp_info.get('turns', 'N/A')
                    print(f"  - Core: {core_name}, Turns: {turns}")

                    if comp_type == "CM_IND" and comp_info.get('inductance_at_freqs'):
                        print("    - 주파수별 인덕턴스 (CM/DM):")
                        for f, L_CM, L_DM in comp_info['inductance_at_freqs']:
                            print(f"      - {f / 1e3:.0f} kHz: CM L = {L_CM:.2f} μH, DM L = {L_DM:.2f} μH")
                    elif comp_type == "DM_IND":
                        if comp_info.get('L_dc') is not None:
                            print(f"    - DM 인덕턴스 (L_DC): {comp_info['L_dc'] * 1e6:.2f} μH")
                        if comp_info.get('L_cm_from_dm_leakage') is not None:
                            print(f"    - DM 리키지 CM 인덕턴스 (L_leakage): {comp_info['L_cm_from_dm_leakage'] * 1e6:.2f} μH")

                    if comp_info.get('SRF') is not None and comp_info['SRF'] != float('inf'):
                        print(f"    - SRF: {comp_info['SRF'] / 1e6:.2f} MHz")
                    elif comp_info.get('SRF') == float('inf'):
                         print("    - SRF: N/A (이상적)")
                    if comp_info.get('ESL') is not None:
                        print(f"    - ESL: {comp_info['ESL'] * 1e9:.2f} nH")
                    if comp_info.get('ESR') is not None:
                        print(f"    - ESR: {comp_info['ESR']:.3f} Ω")
                    if comp_info.get('C_parasitic') is not None:
                        print(f"    - 기생 커패시턴스: {comp_info['C_parasitic'] * 1e12:.2f} pF")

                elif comp_type in ["XCAP", "YCAP"]:
                    cap_type = comp_info.get('cap_type', 'N/A')
                    value_uF = comp_info.get('value_uF', 'N/A')
                    print(f"  - Cap Type: {cap_type}, Value: {value_uF:.2f} uF")
                    if comp_info.get('ESR') is not None:
                        print(f"    - ESR: {comp_info['ESR']:.3f} Ω")
                    if comp_info.get('ESL') is not None:
                        print(f"    - ESL: {comp_info['ESL'] * 1e9:.2f} nH")
                    if comp_info.get('SRF') is not None and comp_info['SRF'] != float('inf'):
                        print(f"    - SRF: {comp_info['SRF'] / 1e6:.2f} MHz")
                    elif comp_info.get('SRF') == float('inf'):
                         print("    - SRF: N/A (이상적)")
                print("-" * 20)

        else:
            print("부품 상세 데이터가 계산 결과에서 발견되지 않았습니다.")


    except Exception as e:
        print(f"Local test 중 오류 발생: {str(e)}")