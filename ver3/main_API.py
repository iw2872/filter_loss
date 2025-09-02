import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
import base64
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional
import uuid
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import plotly.utils

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 세션 미들웨어 임포트
from starlette.middleware.sessions import SessionMiddleware

from pydantic import BaseModel, Field

# filter_insertionloss 모듈이 올바른 위치에 있는지 확인하세요.
from filter_insertionloss import (
    calculate_insertion_loss,
    Inductor,
    Capacitor
)


# --- Pydantic 모델 정의 ---
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
    y_min_db: Optional[float] = Field(None,
                                      description="Manual Y-axis minimum value in dB. If None, auto-calculated.")  # 수정: y_min_db 필드 추가
    y_max_db: Optional[float] = Field(None,
                                      description="Manual Y-axis maximum value in dB. If None, auto-calculated.")  # 수정: y_max_db 필드 추가


# --- FastAPI 애플리케이션 정의 ---
app = FastAPI(
    title="EMI Filter Insertion Loss Calculator",
    description="사용자가 입력한 부품 정보로 EMI 필터의 CM/DM 삽입 손실을 계산하고 그래프로 시각화합니다."
)

# 세션 미들웨어 추가
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


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})


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


@app.post("/calculate_loss", summary="EMI 필터 삽입 손실 계산 (Plotly)")
async def calculate_loss_api(request: FilterTopologyRequest, req: Request):
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

    if not request.topology:
        raise HTTPException(status_code=400, detail="부품 정보가 없습니다. 하나 이상의 부품을 추가하세요.")

    freq = np.logspace(np.log10(start_freq), np.log10(stop_freq), num_points)

    # X축 눈금 설정을 위한 값 생성
    log_min = int(np.floor(np.log10(start_freq)))
    log_max = int(np.ceil(np.log10(stop_freq)))
    tick_vals = [10 ** i for i in range(log_min, log_max + 1)]
    tick_text = []
    for val in tick_vals:
        if val >= 1e9:
            tick_text.append(f'{val / 1e9:.0f}G')
        elif val >= 1e6:
            tick_text.append(f'{val / 1e6:.0f}M')
        elif val >= 1e3:
            tick_text.append(f'{val / 1e3:.0f}k')
        else:
            tick_text.append(str(val))

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
        return {
            "plotly_data": [],
            "plotly_layout": {},
            "component_details_html": f"<p style='color:red;'>필터 계산 중 오류 발생: {str(e)}</p>",
            "message": "Calculation failed. Check your component values."
        }

    # Plotly 그래프 생성
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Common Mode", "Differential Mode"),
        horizontal_spacing=0.1
    )

    # CM 그래프 추가
    fig.add_trace(
        go.Scatter(x=freq, y=IL_cm_db, mode='lines', name='CM Loss',
                   line=dict(color='blue', width=2)),
        row=1, col=1
    )

    # DM 그래프 추가
    fig.add_trace(
        go.Scatter(x=freq, y=IL_dm_db, mode='lines', name='DM Loss',
                   line=dict(color='red', width=2)),
        row=1, col=2
    )

    # --- 수정된 Y축 범위 설정 로직 ---
    # 사용자가 직접 Y축 범위를 입력했는지 확인
    if request.y_min_db is not None and request.y_max_db is not None:
        y_range = [request.y_max_db, request.y_min_db]  # 입력된 값 사용
    else:
        # 입력된 값이 없으면 기존의 자동 계산 로직 사용
        y_min_combined = min(np.min(IL_cm_db), np.min(IL_dm_db))
        y_max_combined = max(np.max(IL_cm_db), np.max(IL_dm_db))
        y_margin = (y_max_combined - y_min_combined) * 0.05
        y_range = [y_max_combined + y_margin, y_min_combined - y_margin]
    # --- 수정된 Y축 범위 설정 로직 끝 ---

    # X축 설정
    shared_xaxis_layout = dict(
        type="log",
        title_text="Frequency (Hz)",
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_text,
        autorange=True,
        showgrid=True,
        gridcolor='#e0e0e0',
        mirror=True,
        linecolor='black',
        linewidth=1,
        ticks='outside',
        title_font=dict(size=14, family="Arial"),
        tickfont=dict(size=12, family="Arial"),
        fixedrange=True
    )

    fig.update_xaxes(**shared_xaxis_layout, row=1, col=1)
    fig.update_xaxes(**shared_xaxis_layout, row=1, col=2)

    # Y축 범위 설정 및 반전 (공유)
    shared_yaxis_layout = dict(
        title_text="Insertion Loss (dB)",
        range=y_range,
        showgrid=True,
        gridcolor='#e0e0e0',
        zeroline=False,
        mirror=True,
        linecolor='black',
        linewidth=1,
        ticks='outside',
        title_font=dict(size=14, family="Arial"),
        tickfont=dict(size=12, family="Arial")
    )
    fig.update_yaxes(**shared_yaxis_layout, row=1, col=1)
    fig.update_yaxes(**shared_yaxis_layout, row=1, col=2)

    # 레이아웃 설정
    fig.update_layout(
        height=600,
        width=1200,
        title_text="EMI Filter Insertion Loss",
        title_x=0.5,
        title_font=dict(size=18, family="Arial", color="black"),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="x unified",
        template="plotly_white"
    )

    # 서브플롯 제목 폰트 설정
    fig.layout.annotations[0].update(font=dict(size=16, family="Arial", color="black"))
    fig.layout.annotations[1].update(font=dict(size=16, family="Arial", color="black"))

    plotly_data = fig.data
    plotly_layout = fig.layout

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
        "plotly_data": json.loads(json.dumps(plotly_data, cls=plotly.utils.PlotlyJSONEncoder)),
        "plotly_layout": json.loads(json.dumps(plotly_layout, cls=plotly.utils.PlotlyJSONEncoder)),
        "component_details_html": format_component_details(all_component_details),
        "message": "Calculation complete. Results are saved in your session."
    }


# --- (이전 코드와 동일한 부분) ---
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