import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
import base64
import pandas as pd
from pathlib import Path
from typing import List, Union
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 세션 미들웨어 임포트
from starlette.middleware.sessions import SessionMiddleware

from pydantic import BaseModel, Field
# 리버스 프록시 : uvicorn main_API_v2:app --host 0.0.0.0 --port 8001 --proxy-headers

# filter_insertionloss 모듈이 올바른 위치에 있는지 확인하세요.
from filter_insertionloss import (
    calculate_insertion_loss,
    Inductor,
    Capacitor
)


# --- Pydantic 모델 정의 (변경 없음) ---
class FilterComponentBaseAPI(BaseModel):
    type: str


class InductorComponentAPI(FilterComponentBaseAPI):
    type: str = Field(..., pattern="^(DM_IND|CM_IND)$")
    core: str
    turns: int


class CapacitorComponentAPI(FilterComponentBaseAPI):
    type: str = Field(..., pattern="^(XCAP|YCAP)$")
    cap_type: str
    value_uF: float


ComponentUnionType = Union[InductorComponentAPI, CapacitorComponentAPI]


class FilterTopologyRequest(BaseModel):
    topology: List[ComponentUnionType]
    start_freq_mhz: float = Field(10e-3, gt=0,
                                  description="Start frequency in MHz (e.g., 0.01 for 10kHz)")
    stop_freq_mhz: float = Field(100.0, gt=0, description="Stop frequency in MHz (e.g., 100 for 100MHz)")
    zin: float = Field(50.0, gt=0, description="Input impedance in Ohms (e.g., 50)")
    zout: float = Field(50.0, gt=0, description="Output impedance in Ohms (e.g., 50)")
    num_frequency_points: int = Field(1000, gt=0, description="Number of frequency points for calculation")


# --- FastAPI 애플리케이션 정의 ---
app = FastAPI(
    title="EMI Filter Insertion Loss Calculator",
    description="사용자가 입력한 부품 정보로 EMI 필터의 CM/DM 삽입 손실을 계산하고 그래프로 시각화합니다."
)

# 세션 미들웨어 추가
# SECRET_KEY는 보안상 노출되지 않는 고유한 값으로 설정해야 합니다.
app.add_middleware(SessionMiddleware, secret_key="your-super-secret-key-that-is-hard-to-guess")

# 계산 결과를 서버 메모리에 저장하는 전역 딕셔너리
session_data_store = {}

# 템플릿과 정적 파일을 설정합니다.
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# CSV 파일에서 코어 이름 불러오기
def load_core_names(filepath: Path) -> List[str]:
    """필수 CSV 파일을 로드하고, 파일이 없으면 서버 시작을 중단합니다."""
    try:
        df = pd.read_csv(filepath)
        if 'core_name' in df.columns:
            return df['core_name'].tolist()
        else:
            raise RuntimeError(f"'{filepath}' 파일에 'core_name' 컬럼이 없습니다.")
    except FileNotFoundError:
        raise RuntimeError(f"필수 코어 데이터베이스 파일 '{filepath}'을(를) 찾을 수 없습니다.")
    except Exception as e:
        raise RuntimeError(f"'{filepath}' 파일을 로드하는 중 오류 발생: {e}")


# 서버 시작 시 필수 데이터를 로드. 실패하면 서버 시작 불가.
try:
    CM_CORE_NAMES = load_core_names(BASE_DIR / 'cm_ANB_core_db.csv')
    DM_CORE_NAMES = load_core_names(BASE_DIR / 'dm_CH_core_db.csv')
except RuntimeError as e:
    print(f"FATAL ERROR: {e}")
    raise e


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """기본 웹 페이지를 제공합니다. `templates/index.html` 파일을 반환합니다."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/get_core_names")
async def get_core_names_api():
    """CSV 파일에서 로드된 CM 및 DM 코어 이름 목록을 반환합니다."""
    return {"cm_cores": CM_CORE_NAMES, "dm_cores": DM_CORE_NAMES}


# --- 다운로드 데이터를 생성하는 헬퍼 함수 ---
def create_s2p_content(freq, s_params, zin, mode: str):
    """S-파라미터 데이터를 기반으로 S2P 파일 내용을 생성합니다."""
    s2p_content = f"# FREQ S RI R {zin:.1f}\n"
    s2p_content += f"! {mode} Mode S-Parameters\n"
    for i in range(len(freq)):
        s2p_content += (
            f"{freq[i]:.6e} {s_params['s11'][i].real:.6e} {s_params['s11'][i].imag:.6e} "
            f"{s_params['s21'][i].real:.6e} {s_params['s21'][i].imag:.6e} "
            f"{s_params['s12'][i].real:.6e} {s_params['s12'][i].imag:.6e} "
            f"{s_params['s22'][i].real:.6e} {s_params['s22'][i].imag:.6e}\n"
        )
    return s2p_content.encode('utf-8')


def create_csv_content(freq, il_db, s_params, mode: str):
    """계산 데이터를 기반으로 CSV 파일 내용을 생성합니다."""
    df = pd.DataFrame({
        'Frequency (Hz)': freq,
        f'{mode}_IL (dB)': il_db,
        f'S11_{mode}_Real': s_params['s11'].real,
        f'S11_{mode}_Imag': s_params['s11'].imag,
        f'S12_{mode}_Real': s_params['s12'].real,
        f'S12_{mode}_Imag': s_params['s12'].imag,
        f'S21_{mode}_Real': s_params['s21'].real,
        f'S21_{mode}_Imag': s_params['s21'].imag,
        f'S22_{mode}_Real': s_params['s22'].real,
        f'S22_{mode}_Imag': s_params['s22'].imag
    })
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode('utf-8')


def format_component_details(all_component_details):
    """부품 상세 정보를 HTML 문자열로 포맷팅합니다."""
    component_details_html = ""
    if not all_component_details:
        return ""

    component_details_html += "<h3>모든 부품의 상세 정보:</h3>"
    for i, comp_info in enumerate(all_component_details):
        comp_type = comp_info.get('component_type', 'N/A')
        component_details_html += f"<h4>부품 {i + 1} (유형: {comp_type})"
        if comp_type in ["CM_IND", "DM_IND"]:
            core_name = comp_info.get('core_name', 'N/A')
            turns = comp_info.get('turns', 'N/A')
            component_details_html += f" - 코어: {core_name}, 권선: {turns} 턴"
            component_details_html += "</h4><ul>"
            if comp_type == "CM_IND" and comp_info.get('inductance_at_freqs'):
                component_details_html += "<li><strong>주파수별 인덕턴스:</strong></li>"
                for f_hz, L_CM_uH, L_DM_uH in comp_info['inductance_at_freqs']:
                    component_details_html += f"<li>&nbsp;&nbsp;&nbsp;- {f_hz / 1e3:.0f} kHz: CM L = {L_CM_uH:.2f} μH, DM L = {L_DM_uH:.2f} μH</li>"
            elif comp_type == "DM_IND":
                L_dc_val = comp_info.get('L_dc')
                if L_dc_val is not None:
                    component_details_html += f"<li><strong>DM 인덕턴스 (L_DC):</strong> {L_dc_val * 1e6:.2f} μH</li>"
                L_cm_from_dm_leakage_val = comp_info.get('L_cm_from_dm_leakage')
                if L_cm_from_dm_leakage_val is not None:
                    component_details_html += f"<li><strong>DM 리키지 CM 인덕턴스 (L_leakage):</strong> {L_cm_from_dm_leakage_val * 1e6:.2f} μH</li>"
        elif comp_type in ["XCAP", "YCAP"]:
            cap_type = comp_info.get('cap_type', 'N/A')
            value_uF = comp_info.get('value_uF', 'N/A')
            component_details_html += f" - 타입: {cap_type}, 용량: {value_uF:.2f} μF"
            component_details_html += "</h4><ul>"
        else:
            component_details_html += "</h4><ul>"
        esr_val = comp_info.get('ESR')
        if esr_val is not None:
            component_details_html += f"<li><strong>ESR:</strong> {esr_val:.3f} Ω</li>"
        esl_val = comp_info.get('ESL')
        if esl_val is not None:
            component_details_html += f"<li><strong>ESL:</strong> {esl_val * 1e9:.2f} nH</li>"
        srf_val = comp_info.get('SRF')
        if srf_val is not None:
            if srf_val == float('inf'):
                component_details_html += f"<li><strong>SRF:</strong> N/A (이상적)</li>"
            else:
                component_details_html += f"<li><strong>SRF:</strong> {srf_val / 1e6:.2f} MHz</li>"
        c_parasitic_val = comp_info.get('C_parasitic')
        if c_parasitic_val is not None:
            component_details_html += f"<li><strong>기생 커패시턴스:</strong> {c_parasitic_val * 1e12:.2f} pF</li>"
        component_details_html += "</ul>"
        component_details_html += "<hr>"
    return component_details_html


@app.post("/calculate_loss", summary="EMI 필터 삽입 손실 계산")
async def calculate_loss_api(request: FilterTopologyRequest, req: Request):
    """
    필터 토폴로지 및 주파수 범위에 대한 CM 및 DM 삽입 손실을 계산하고,
    결과 데이터를 세션에 저장합니다.
    """
    start_freq = request.start_freq_mhz * 1e6
    stop_freq = request.stop_freq_mhz * 1e6
    num_points = request.num_frequency_points
    zin, zout = request.zin, request.zout

    if start_freq >= stop_freq or start_freq <= 0 or stop_freq <= 0:
        raise HTTPException(status_code=400, detail="유효한 주파수 범위를 입력하세요.")
    if zin <= 0 or zout <= 0:
        raise HTTPException(status_code=400, detail="입출력 임피던스는 양수여야 합니다.")
    if num_points <= 0:
        raise HTTPException(status_code=400, detail="주파수 개수는 양수여야 합니다.")

    freq = np.logspace(np.log10(start_freq), np.log10(stop_freq), num_points)
    converted_topology = []
    for comp_data in request.topology:
        if comp_data.type in ["DM_IND", "CM_IND"]:
            converted_topology.append(Inductor(type=comp_data.type, core=comp_data.core, turns=comp_data.turns))
        elif comp_data.type in ["XCAP", "YCAP"]:
            converted_topology.append(
                Capacitor(type=comp_data.type, cap_type=comp_data.cap_type, value_uF=comp_data.value_uF))
        else:
            raise HTTPException(status_code=400, detail=f"알 수 없는 부품 타입: {comp_data.type}")

    try:
        IL_cm_db, s11_cm, s12_cm, s21_cm, s22_cm, all_component_details = calculate_insertion_loss(freq,
                                                                                                   converted_topology,
                                                                                                   mode="CM", Zin=zin,
                                                                                                   Zout=zout)
        IL_dm_db, s11_dm, s12_dm, s21_dm, s22_dm, _ = calculate_insertion_loss(freq, converted_topology, mode="DM",
                                                                               Zin=zin, Zout=zout)
    except Exception as e:
        print(f"Error during filter calculation: {e}")
        raise HTTPException(status_code=400, detail=f"필터 계산 중 오류 발생: {str(e)}")

    plt.rcParams['axes.unicode_minus'] = False
    if plt.rcParams['font.family'][0] == 'sans-serif':
        plt.rc('font', family='Malgun Gothic')
    else:
        plt.rc('font', family='DejaVu Sans')
    plt.rc('mathtext', fontset='cm')

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    def log_formatter(x, pos):
        if x >= 1e9:
            return f'{x * 1e-9:.0f}G'
        elif x >= 1e6:
            return f'{x * 1e-6:.0f}M'
        elif x >= 1e3:
            return f'{x * 1e-3:.0f}k'
        else:
            return f'{x:.0f}'

    # y축 범위를 수동으로 설정합니다.
    # 계산된 삽입 손실 값의 최소값을 찾습니다.
    y_max = np.max([IL_cm_db, IL_dm_db])

    # 그래프의 y축을 반전된 형태로 설정 (최소값부터 0dB까지)
    axs[0].set_ylim(y_max + 10, -5)
    axs[1].set_ylim(y_max + 10, -5)

    formatter = ticker.FuncFormatter(log_formatter)
    major_locator = ticker.LogLocator(base=10.0, numticks=None)
    minor_locator = ticker.NullLocator()

    axs[0].semilogx(freq, IL_cm_db, label="CM Insertion Loss", color='blue')
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Insertion Loss (dB)")
    axs[0].set_title("EMI Filter Insertion Loss - CM Mode")
    axs[0].grid(True, which='both', linestyle='--')
    axs[0].legend()
    #axs[0].invert_yaxis()
    axs[0].xaxis.set_major_locator(major_locator)
    axs[0].xaxis.set_major_formatter(formatter)
    axs[0].xaxis.set_minor_locator(minor_locator)

    axs[1].semilogx(freq, IL_dm_db, label="DM Insertion Loss", color='red')
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_title("EMI Filter Insertion Loss - DM Mode")
    axs[1].grid(True, which='both', linestyle='--')
    axs[1].legend()
    #axs[1].invert_yaxis()
    axs[1].xaxis.set_major_locator(major_locator)
    axs[1].xaxis.set_major_formatter(formatter)
    axs[1].xaxis.set_minor_locator(minor_locator)



    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # 복소수 NumPy 배열을 JSON 직렬화 가능한 리스트로 변환
    s_params_cm = {
        "s11": [[s.real, s.imag] for s in s11_cm],
        "s12": [[s.real, s.imag] for s in s12_cm],
        "s21": [[s.real, s.imag] for s in s21_cm],
        "s22": [[s.real, s.imag] for s in s22_cm]
    }
    s_params_dm = {
        "s11": [[s.real, s.imag] for s in s11_dm],
        "s12": [[s.real, s.imag] for s in s12_dm],
        "s21": [[s.real, s.imag] for s in s21_dm],
        "s22": [[s.real, s.imag] for s in s22_dm]
    }

    # 계산 결과를 서버 메모리(전역 딕셔너리)에 저장합니다.
    session_id = req.session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        req.session['session_id'] = session_id

    session_data_store[session_id] = {
        "s_params_cm": s_params_cm,
        "s_params_dm": s_params_dm,
        "freq": freq.tolist(),
        "il_cm_db": IL_cm_db.tolist(),
        "il_dm_db": IL_dm_db.tolist(),
        "zin": zin,
    }

    return {
        "cm_dm_graph_base64": image_base64,
        "component_details_html": format_component_details(all_component_details),
        "message": "Calculation complete. Results are saved in your session."
    }


# --- 다운로드 엔드포인트를 GET 요청을 받도록 변경 ---
def get_session_data_or_raise_exception(req: Request):
    session_id = req.session.get('session_id')
    if not session_id or session_id not in session_data_store:
        raise HTTPException(status_code=400, detail="계산 결과가 없습니다. 먼저 계산을 실행하세요.")
    return session_data_store[session_id]


@app.get("/download_s2p_cm", summary="CM S2P 파일 다운로드")
async def download_s2p_cm_api(req: Request):
    data = get_session_data_or_raise_exception(req)
    freq = np.array(data['freq'])
    s_params_cm = {
        "s11": np.array([complex(r, i) for r, i in data['s_params_cm']['s11']]),
        "s12": np.array([complex(r, i) for r, i in data['s_params_cm']['s12']]),
        "s21": np.array([complex(r, i) for r, i in data['s_params_cm']['s21']]),
        "s22": np.array([complex(r, i) for r, i in data['s_params_cm']['s22']])
    }
    s2p_content = create_s2p_content(freq, s_params_cm, data['zin'], "CM")
    return Response(content=s2p_content, media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=filter_s_parameters_cm.s2p"})


@app.get("/download_s2p_dm", summary="DM S2P 파일 다운로드")
async def download_s2p_dm_api(req: Request):
    data = get_session_data_or_raise_exception(req)
    freq = np.array(data['freq'])
    s_params_dm = {
        "s11": np.array([complex(r, i) for r, i in data['s_params_dm']['s11']]),
        "s12": np.array([complex(r, i) for r, i in data['s_params_dm']['s12']]),
        "s21": np.array([complex(r, i) for r, i in data['s_params_dm']['s21']]),
        "s22": np.array([complex(r, i) for r, i in data['s_params_dm']['s22']])
    }
    s2p_content = create_s2p_content(freq, s_params_dm, data['zin'], "DM")
    return Response(content=s2p_content, media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=filter_s_parameters_dm.s2p"})


@app.get("/download_csv_cm", summary="CM CSV 파일 다운로드")
async def download_csv_cm_api(req: Request):
    data = get_session_data_or_raise_exception(req)
    freq = np.array(data['freq'])
    il_cm_db = np.array(data['il_cm_db'])
    s_params_cm = {
        "s11": np.array([complex(r, i) for r, i in data['s_params_cm']['s11']]),
        "s12": np.array([complex(r, i) for r, i in data['s_params_cm']['s12']]),
        "s21": np.array([complex(r, i) for r, i in data['s_params_cm']['s21']]),
        "s22": np.array([complex(r, i) for r, i in data['s_params_cm']['s22']])
    }
    csv_content = create_csv_content(freq, il_cm_db, s_params_cm, "CM")
    return Response(content=csv_content, media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=filter_data_cm.csv"})


@app.get("/download_csv_dm", summary="DM CSV 파일 다운로드")
async def download_csv_dm_api(req: Request):
    data = get_session_data_or_raise_exception(req)
    freq = np.array(data['freq'])
    il_dm_db = np.array(data['il_dm_db'])
    s_params_dm = {
        "s11": np.array([complex(r, i) for r, i in data['s_params_dm']['s11']]),
        "s12": np.array([complex(r, i) for r, i in data['s_params_dm']['s12']]),
        "s21": np.array([complex(r, i) for r, i in data['s_params_dm']['s21']]),
        "s22": np.array([complex(r, i) for r, i in data['s_params_dm']['s22']])
    }
    csv_content = create_csv_content(freq, il_dm_db, s_params_dm, "DM")
    return Response(content=csv_content, media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=filter_data_dm.csv"})


