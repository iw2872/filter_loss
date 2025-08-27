# main_API.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
import base64
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Union

from filter_insertionloss import (
    calculate_insertion_loss,
    FilterComponent,
    Inductor,
    Capacitor
)


# --- Pydantic 모델 정의 (FastAPI 입력 스키마) ---
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


# CSV 파일에서 코어 이름 불러오기
def load_core_names(filepath: str) -> List[str]:
    try:
        df = pd.read_csv(filepath)
        if 'core_name' in df.columns:
            return df['core_name'].tolist()
        else:
            print(f"Warning: 'core_name' column not found in {filepath}. Returning empty list.")
            return []
    except FileNotFoundError:
        print(f"Error: Core file not found at {filepath}. Please ensure {filepath} exists.")
        return []
    except Exception as e:
        print(f"Error loading core names from {filepath}: {e}. Returning empty list.")
        return []


CM_CORE_NAMES = load_core_names('./cm_ANB_core_db.csv')
DM_CORE_NAMES = load_core_names('./dm_CH_core_db.csv')

# 마지막 계산된 S-파라미터 데이터를 저장할 전역 변수 (CM, DM 분리)
last_calculated_s_params_cm = None
last_calculated_s_params_dm = None
last_calculated_il_cm = None
last_calculated_il_dm = None
last_calculated_freq = None
last_calculated_impedances = None
# 모든 부품 상세 정보를 저장할 전역 변수 추가
last_calculated_component_details = None


@app.get("/", response_class=HTMLResponse, summary="Welcome Page")
async def read_root():
    """기본 웹 페이지를 제공합니다. 사용자가 필터 정보를 입력할 수 있는 간단한 폼을 포함합니다."""
    # `result.inductor_details_html`을 `result.component_details_html`로 변경
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EMI Filter Insertion Loss Calculator</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            form { border: 1px solid #ccc; padding: 20px; border-radius: 8px; max-width: 600px; margin-bottom: 30px; }
            .component-group { border: 1px solid #eee; padding: 15px; margin-bottom: 15px; border-radius: 5px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="text"], input[type="number"], select {
                width: calc(100% - 22px); padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px;
            }
            button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            #results img { max-width: 100%; height: auto; display: block; margin-top: 20px; border: 1px solid #ddd;}
            .add-button { background-color: #28a745; margin-top: 10px; }
            .add-button:hover { background-color: #218838; }
            .remove-button { background-color: #dc3545; margin-left: 10px; }
            .remove-button:hover { background-color: #c82333; }
            .hidden { display: none; }

            .download-button {
                display: inline-block;
                padding: 10px 15px;
                margin: 5px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                text-decoration: none;
                font-weight: bold;
                cursor: pointer;
                transition: background-color 0.2s ease;
            }
            .download-button:hover {
                background-color: #0056b3;
            }
            .download-button.csv {
                background-color: #28a745;
            }
            .download-button.csv:hover {
                background-color: #218838;
            }

            .container {
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }

            .left-panel {
                flex: 1;
                min-width: 400px;
                max-width: 600px;
            }

            .right-panel {
                flex: 2;
                min-width: 500px;
            }

            #results {
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #f9f9f9;
            }
            .top-calculate-button {
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <h1>EMI Filter Insertion Loss Calculator V.1.0</h1>

        <div class="container">
            <div class="left-panel">
                <form id="filterForm">
                    <button type="submit" class="top-calculate-button">계산 및 그래프 보기</button>
                    <hr>

                    <div id="components-container">
                        <div class="component-group" id="component-0">
                            <h3>부품 1</h3>
                            <label for="type-0">부품 타입:</label>
                            <select id="type-0" name="components[0].type" onchange="toggleComponentFields(0)">
                                <option value="">선택하세요</option>
                                <option value="CM_IND">CM 인덕터</option>
                                <option value="DM_IND">DM 인덕터</option>
                                <option value="XCAP">X 커패시터</option>
                                <option value="YCAP">Y 커패시터</option>
                            </select>

                            <div class="inductor-fields hidden" id="inductor-fields-0">
                                <label for="core-0">코어:</label>
                                <select id="core-0" name="components[0].core">
                                    <option value="">코어를 선택하세요</option>
                                    </select>
                                <label for="turns-0">권선 수:</label>
                                <input type="number" id="turns-0" name="components[0].turns" placeholder="예: 7">
                            </div>

                            <div class="capacitor-fields hidden" id="capacitor-fields-0">
                                <label for="cap_type-0">커패시터 타입:</label>
                                <select id="cap_type-0" name="components[0].cap_type">
                                    <option value="FILM">FILM</option>
                                    <option value="CERAMIC">CERAMIC</option>
                                </select>
                                <label for="value_uF-0">용량 (uF):</label>
                                <input type="number" step="any" id="value_uF-0" name="components[0].value_uF" placeholder="예: 0.1">
                            </div>
                            <button type="button" class="remove-button hidden" onclick="removeComponent(0)">부품 제거</button>
                        </div>
                    </div>
                    <button type="button" class="add-button" onclick="addComponent()">부품 추가</button><br><br>

                    <div style="border: 1px solid #eee; padding: 15px; border-radius: 5px; margin-top: 20px;">
                        <h3>주파수 범위 설정 (MHz)</h3>
                        <label for="start_freq_mhz">시작 주파수 (MHz):</label>
                        <input type="number" step="any" id="start_freq_mhz" name="start_freq_mhz" value="0.01" min="0.001" max="1000"><br>
                        <label for="stop_freq_mhz">종료 주파수 (MHz):</label>
                        <input type="number" step="any" id="stop_freq_mhz" name="stop_freq_mhz" value="100" min="0.001" max="1000"><br>
                        <label for="num_frequency_points">주파수 개수:</label> <input type="number" id="num_frequency_points" name="num_frequency_points" value="500" min="10" max="5000"><br> </div>
                    <br>
                    <div style="border: 1px solid #eee; padding: 15px; border-radius: 5px; margin-top: 20px;">
                        <h3>입출력 임피던스 설정 (Ω)</h3>
                        <label for="zin">입력 임피던스 (Ω):</label>
                        <input type="number" step="any" id="zin" name="zin" value="50" min="1"><br>
                        <label for="zout">출력 임피던스 (Ω):</label>
                        <input type="number" step="any" id="zout" name="zout" value="50" min="1"><br>
                    </div>
                    <br>
                </form>
            </div>
            <div class="right-panel">
                <div id="results">
                    </div>
            </div>
        </div>

        <script>
            let componentCount = 1;
            let cmCoreNames = [];
            let dmCoreNames = [];

            async function loadInitialCoreNames() {
                try {
                    const response = await fetch('/get_core_names');
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const data = await response.json();
                    cmCoreNames = data.cm_cores;
                    dmCoreNames = data.dm_cores;
                    console.log("CM Cores:", cmCoreNames);
                    console.log("DM Cores:", dmCoreNames);
                    updateCoreDropdown(0, document.getElementById('type-0').value);
                } catch (error) {
                    console.error("Failed to load core names:", error);
                    alert("코어 이름을 로드하는 데 실패했습니다. 서버 로그를 확인하세요.");
                }
            }

            function updateCoreDropdown(index, componentType) {
                const coreSelect = document.getElementById(`core-${index}`);
                if (!coreSelect) return;

                while (coreSelect.options.length > 1) {
                    coreSelect.remove(1);
                }

                let coresToUse = [];
                if (componentType === 'CM_IND') {
                    coresToUse = cmCoreNames;
                } else if (componentType === 'DM_IND') {
                    coresToUse = dmCoreNames;
                }

                coresToUse.forEach(core => {
                    const option = document.createElement('option');
                    option.value = core;
                    option.textContent = core;
                    coreSelect.appendChild(option);
                });
            }

            function toggleComponentFields(index) {
                const type = document.getElementById(`type-${index}`).value;
                const inductorFields = document.getElementById(`inductor-fields-${index}`);
                const capacitorFields = document.getElementById(`capacitor-fields-${index}`);

                inductorFields.classList.add('hidden');
                capacitorFields.classList.add('hidden');

                if (type === 'CM_IND' || type === 'DM_IND') {
                    inductorFields.classList.remove('hidden');
                    updateCoreDropdown(index, type);
                } else if (type === 'XCAP' || type === 'YCAP') {
                    capacitorFields.classList.remove('hidden');
                }
            }

            function addComponent() {
                const container = document.getElementById('components-container');
                const newComponentIndex = componentCount++;
                const newComponentHtml = `
                    <div class="component-group" id="component-${newComponentIndex}">
                        <h3>부품 ${newComponentIndex + 1}</h3>
                        <label for="type-${newComponentIndex}">부품 타입:</label>
                        <select id="type-${newComponentIndex}" name="components[${newComponentIndex}].type" onchange="toggleComponentFields(${newComponentIndex})">
                            <option value="">선택하세요</option>
                            <option value="CM_IND">CM 인덕터</option>
                            <option value="DM_IND">DM 인덕터</option>
                            <option value="XCAP">X 커패시터</option>
                            <option value="YCAP">Y 커패시터</option>
                        </select>

                        <div class="inductor-fields hidden" id="inductor-fields-${newComponentIndex}">
                            <label for="core-${newComponentIndex}">코어:</label>
                            <select id="core-${newComponentIndex}" name="components[${newComponentIndex}].core">
                                <option value="">코어를 선택하세요</option>
                                </select>
                            <label for="turns-${newComponentIndex}">권선 수:</label>
                            <input type="number" id="turns-${newComponentIndex}" name="components[${newComponentIndex}].turns" placeholder="예: 7">
                        </div>

                        <div class="capacitor-fields hidden" id="capacitor-fields-${newComponentIndex}">
                            <label for="cap_type-${newComponentIndex}">커패시터 타입:</label>
                            <select id="cap_type-${newComponentIndex}" name="components[${newComponentIndex}].cap_type">
                                <option value="FILM">FILM</option>
                                <option value="CERAMIC">CERAMIC</option>
                            </select>
                            <label for="value_uF-${newComponentIndex}">용량 (uF):</label>
                            <input type="number" step="any" id="value_uF-${newComponentIndex}" name="components[${newComponentIndex}].value_uF" placeholder="예: 0.1">
                        </div>
                        <button type="button" class="remove-button" onclick="removeComponent(${newComponentIndex})">부품 제거</button>
                    </div>
                `;
                container.insertAdjacentHTML('beforeend', newComponentHtml);
                toggleComponentFields(newComponentIndex);
            }

            function removeComponent(index) {
                const componentToRemove = document.getElementById(`component-${index}`);
                if (componentToRemove) {
                    componentToRemove.remove();
                }
            }

            document.getElementById('filterForm').addEventListener('submit', async function(event) {
                event.preventDefault();

                const formData = new FormData(event.target);
                const components = [];
                let requestBody = {};

                for (let [name, value] of formData.entries()) {
                    const compMatch = name.match(/components\[(\d+)\]\.(.+)/);
                    if (compMatch) {
                        const index = parseInt(compMatch[1]);
                        const field = compMatch[2];

                        if (!components[index]) {
                            components[index] = {};
                        }
                        if (['turns', 'value_uF'].includes(field)) {
                            components[index][field] = parseFloat(value);
                        } else {
                            components[index][field] = value;
                        }
                    } else {
                        if (['start_freq_mhz', 'stop_freq_mhz', 'zin', 'zout', 'num_frequency_points'].includes(name)) {
                            requestBody[name] = parseFloat(value);
                        } else {
                            requestBody[name] = value; // 다른 필드도 포함될 수 있도록 추가
                        }
                    }
                }

                const validComponents = components.filter(comp => comp && comp.type);
                requestBody.topology = validComponents;

                try {
                    const response = await fetch('/calculate_loss', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(requestBody)
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        alert(`Error: ${errorData.detail || response.statusText}`);
                        return;
                    }

                    const result = await response.json();
                    const resultsDiv = document.getElementById('results');
                    resultsDiv.innerHTML = `
                        <h2>계산 결과</h2>
                        <img src="data:image/png;base64,${result.cm_dm_graph_base64}" alt="CM/DM Insertion Loss Graph">
                        <div style="margin-top: 20px; text-align: center;">
                            <a href="/download_s2p_cm" target="_blank" class="download-button">CM S2P 파일 다운로드</a>
                            <a href="/download_s2p_dm" target="_blank" class="download-button">DM S2P 파일 다운로드</a>
                            <a href="/download_csv_cm" target="_blank" class="download-button csv">CM CSV 파일 다운로드</a>
                            <a href="/download_csv_dm" target="_blank" class="download-button csv">DM CSV 파일 다운로드</a>
                        </div>
                        ${result.component_details_html || ''} `; } catch (error) {
                    console.error('Error:', error);
                    alert('An error occurred while calculating insertion loss.');
                }
            });

            loadInitialCoreNames();
        </script>
    </body>
    </html>
    """


@app.get("/get_core_names")
async def get_core_names_api():
    """CSV 파일에서 로드된 CM 및 DM 코어 이름 목록을 반환합니다."""
    return {"cm_cores": CM_CORE_NAMES, "dm_cores": DM_CORE_NAMES}


@app.post("/calculate_loss", summary="EMI 필터 삽입 손실 계산")
async def calculate_loss_api(request: FilterTopologyRequest):
    """
    제공된 필터 토폴로지 및 주파수 범위에 대한 CM 및 DM 삽입 손실을 계산하고,
    결과 그래프 이미지를 Base64 문자열과 함께 모든 부품의 상세 정보를 반환합니다.
    """
    start_freq = request.start_freq_mhz * 1e6
    stop_freq = request.stop_freq_mhz * 1e6
    num_points = request.num_frequency_points

    zin = request.zin
    zout = request.zout

    if start_freq >= stop_freq:
        raise HTTPException(status_code=400, detail="시작 주파수는 끝 주파수보다 작아야 합니다.")
    if start_freq <= 0 or stop_freq <= 0:
        raise HTTPException(status_code=400, detail="주파수는 양수여야 합니다.")
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
        # filter_insertionloss에서 all_component_details를 반환하도록 수정되었으므로, 이를 받습니다.
        # 부품 상세 정보는 CM 계산에서 반환되는 데이터를 사용합니다.
        IL_cm_db, s11_cm, s12_cm, s21_cm, s22_cm, all_component_details = calculate_insertion_loss(freq,
                                                                                                      converted_topology,
                                                                                                      mode="CM",
                                                                                                      Zin=zin,
                                                                                                      Zout=zout)
        # DM 계산에서도 all_component_details를 받지만, 위에서 이미 받았으므로 '_'로 무시합니다.
        IL_dm_db, s11_dm, s12_dm, s21_dm, s22_dm, _ = calculate_insertion_loss(freq, converted_topology, mode="DM",
                                                                               Zin=zin, Zout=zout)
    except Exception as e:
        print(f"Error during filter calculation: {e}")
        raise HTTPException(status_code=400, detail=f"필터 계산 중 오류 발생: {str(e)}")

    # --- 그래프 생성 ---
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지 (Windows/Linux)
    # 한글 폰트 설정 (Windows/Linux에서 'Malgun Gothic'이 없으면 'DejaVu Sans' 사용)
    # macOS의 경우 'AppleGothic' 또는 'NanumGothic' 등을 사용할 수 있습니다.
    if plt.rcParams['font.family'][0] == 'sans-serif':
        plt.rc('font', family='Malgun Gothic')
    else:
        plt.rc('font', family='DejaVu Sans')
    plt.rc('mathtext', fontset='cm')  # LaTeX 스타일 폰트 유지 (선택 사항)


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

    formatter = ticker.FuncFormatter(log_formatter)

    major_locator = ticker.LogLocator(base=10.0, numticks=None)
    minor_locator = ticker.NullLocator()

    axs[0].semilogx(freq, IL_cm_db, label="CM Insertion Loss", color='blue')
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Insertion Loss (dB)")
    axs[0].set_title("EMI Filter Insertion Loss - CM Mode")
    axs[0].grid(True, which='both', linestyle='--')
    axs[0].legend()
    axs[0].invert_yaxis()
    axs[0].xaxis.set_major_locator(major_locator)
    axs[0].xaxis.set_major_formatter(formatter)
    axs[0].xaxis.set_minor_locator(minor_locator)

    axs[1].semilogx(freq, IL_dm_db, label="DM Insertion Loss", color='red')
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_title("EMI Filter Insertion Loss - DM Mode")
    axs[1].grid(True, which='both', linestyle='--')
    axs[1].legend()
    axs[1].invert_yaxis()
    axs[1].xaxis.set_major_locator(major_locator)
    axs[1].xaxis.set_major_formatter(formatter)
    axs[1].xaxis.set_minor_locator(minor_locator)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # S2P 및 CSV 파일 생성을 위한 데이터 저장 (CM, DM 분리)
    global last_calculated_s_params_cm, last_calculated_s_params_dm, \
        last_calculated_il_cm, last_calculated_il_dm, \
        last_calculated_freq, last_calculated_impedances, last_calculated_component_details

    last_calculated_freq = freq.tolist()
    last_calculated_impedances = {"zin": zin, "zout": zout}

    last_calculated_il_cm = IL_cm_db.tolist()
    last_calculated_il_dm = IL_dm_db.tolist()

    last_calculated_s_params_cm = {
        "s11": [[s.real, s.imag] for s in s11_cm],
        "s12": [[s.real, s.imag] for s in s12_cm],
        "s21": [[s.real, s.imag] for s in s21_cm],
        "s22": [[s.real, s.imag] for s in s22_cm]
    }
    last_calculated_s_params_dm = {
        "s11": [[s.real, s.imag] for s in s11_dm],
        "s12": [[s.real, s.imag] for s in s12_dm],
        "s21": [[s.real, s.imag] for s in s21_dm],
        "s22": [[s.real, s.imag] for s in s22_dm]
    }

    # 모든 부품의 상세 정보를 HTML 문자열로 변환
    component_details_html = ""
    if all_component_details:
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
                component_details_html += "</h4><ul>" # Unknown type, just close h4 and open ul

            # 공통적으로 있을 수 있는 기생 파라미터 (단위 변환 및 None 체크)
            esr_val = comp_info.get('ESR')
            if esr_val is not None:
                component_details_html += f"<li><strong>ESR:</strong> {esr_val:.3f} Ω</li>" # 소수점 셋째 자리까지

            esl_val = comp_info.get('ESL')
            if esl_val is not None:
                component_details_html += f"<li><strong>ESL:</strong> {esl_val * 1e9:.2f} nH</li>" # H -> nH

            srf_val = comp_info.get('SRF')
            if srf_val is not None:
                if srf_val == float('inf'):
                    component_details_html += f"<li><strong>SRF:</strong> N/A (이상적)</li>"
                else:
                    component_details_html += f"<li><strong>SRF:</strong> {srf_val / 1e6:.2f} MHz</li>" # Hz -> MHz

            c_parasitic_val = comp_info.get('C_parasitic')
            if c_parasitic_val is not None:
                component_details_html += f"<li><strong>기생 커패시턴스:</strong> {c_parasitic_val * 1e12:.2f} pF</li>" # F -> pF

            component_details_html += "</ul>"
            component_details_html += "<hr>" # 각 부품 정보 섹션 구분

    # 모든 부품 상세 정보를 전역 변수에 저장
    last_calculated_component_details = all_component_details

    return {"cm_dm_graph_base64": image_base64, "component_details_html": component_details_html}


# --- CM S2P 파일 다운로드 엔드포인트 ---
@app.get("/download_s2p_cm", summary="CM S2P 파일 다운로드")
async def download_s2p_cm_api():
    """
    마지막으로 계산된 CM S-파라미터 데이터를 S2P 파일 형식으로 반환합니다.
    """
    global last_calculated_s_params_cm, last_calculated_freq, last_calculated_impedances

    if last_calculated_s_params_cm is None or last_calculated_freq is None or last_calculated_impedances is None:
        raise HTTPException(status_code=404, detail="CM S-파라미터 데이터가 없습니다. 먼저 삽입 손실을 계산하세요.")

    freq = np.array(last_calculated_freq)
    zin = last_calculated_impedances["zin"]

    s11_cm = np.array([complex(r, i) for r, i in last_calculated_s_params_cm["s11"]])
    s12_cm = np.array([complex(r, i) for r, i in last_calculated_s_params_cm["s12"]])
    s21_cm = np.array([complex(r, i) for r, i in last_calculated_s_params_cm["s21"]])
    s22_cm = np.array([complex(r, i) for r, i in last_calculated_s_params_cm["s22"]])

    s2p_content_cm = f"# FREQ S RI R {zin:.1f}\n"
    s2p_content_cm += "! CM Mode S-Parameters\n"
    for i in range(len(freq)):
        s2p_content_cm += (
            f"{freq[i]:.6e} {s11_cm[i].real:.6e} {s11_cm[i].imag:.6e} "
            f"{s21_cm[i].real:.6e} {s21_cm[i].imag:.6e} "
            f"{s12_cm[i].real:.6e} {s12_cm[i].imag:.6e} "
            f"{s22_cm[i].real:.6e} {s22_cm[i].imag:.6e}\n"
        )

    return StreamingResponse(
        io.BytesIO(s2p_content_cm.encode('utf-8')),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=filter_s_parameters_cm.s2p"}
    )


# --- DM S2P 파일 다운로드 엔드포인트 ---
@app.get("/download_s2p_dm", summary="DM S2P 파일 다운로드")
async def download_s2p_dm_api():
    """
    마지막으로 계산된 DM S-파라미터 데이터를 S2P 파일 형식으로 반환합니다.
    """
    global last_calculated_s_params_dm, last_calculated_freq, last_calculated_impedances

    if last_calculated_s_params_dm is None or last_calculated_freq is None or last_calculated_impedances is None:
        raise HTTPException(status_code=404, detail="DM S-파라미터 데이터가 없습니다. 먼저 삽입 손실을 계산하세요.")

    freq = np.array(last_calculated_freq)
    zin = last_calculated_impedances["zin"]

    s11_dm = np.array([complex(r, i) for r, i in last_calculated_s_params_dm["s11"]])
    s12_dm = np.array([complex(r, i) for r, i in last_calculated_s_params_dm["s12"]])
    s21_dm = np.array([complex(r, i) for r, i in last_calculated_s_params_dm["s21"]])
    s22_dm = np.array([complex(r, i) for r, i in last_calculated_s_params_dm["s22"]])

    s2p_content_dm = f"# FREQ S RI R {zin:.1f}\n"
    s2p_content_dm += "! DM Mode S-Parameters\n"
    for i in range(len(freq)):
        s2p_content_dm += (
            f"{freq[i]:.6e} {s11_dm[i].real:.6e} {s11_dm[i].imag:.6e} "
            f"{s21_dm[i].real:.6e} {s21_dm[i].imag:.6e} "
            f"{s12_dm[i].real:.6e} {s12_dm[i].imag:.6e} "
            f"{s22_dm[i].real:.6e} {s22_dm[i].imag:.6e}\n"
        )

    return StreamingResponse(
        io.BytesIO(s2p_content_dm.encode('utf-8')),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=filter_s_parameters_dm.s2p"}
    )


# --- CM CSV 파일 다운로드 엔드포인트 ---
@app.get("/download_csv_cm", summary="CM CSV 파일 다운로드")
async def download_csv_cm_api():
    """
    마지막으로 계산된 CM 삽입 손실 및 S-파라미터 데이터를 CSV 파일 형식으로 반환합니다.
    """
    global last_calculated_s_params_cm, last_calculated_il_cm, last_calculated_freq

    if last_calculated_s_params_cm is None or last_calculated_il_cm is None or last_calculated_freq is None:
        raise HTTPException(status_code=404, detail="CM 데이터가 없습니다. 먼저 삽입 손실을 계산하세요.")

    freq = np.array(last_calculated_freq)
    il_cm_db = np.array(last_calculated_il_cm)
    s11_cm = np.array([complex(r, i) for r, i in last_calculated_s_params_cm["s11"]])
    s12_cm = np.array([complex(r, i) for r, i in last_calculated_s_params_cm["s12"]])
    s21_cm = np.array([complex(r, i) for r, i in last_calculated_s_params_cm["s21"]])
    s22_cm = np.array([complex(r, i) for r, i in last_calculated_s_params_cm["s22"]])

    # pandas DataFrame을 사용하여 CSV 생성
    df = pd.DataFrame({
        'Frequency (Hz)': freq,
        'CM_IL (dB)': il_cm_db,
        'S11_CM_Real': s11_cm.real,
        'S11_CM_Imag': s11_cm.imag,
        'S12_CM_Real': s12_cm.real,
        'S12_CM_Imag': s12_cm.imag,
        'S21_CM_Real': s21_cm.real,
        'S21_CM_Imag': s21_cm.imag,
        'S22_CM_Real': s22_cm.real,
        'S22_CM_Imag': s22_cm.imag
    })

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=filter_data_cm.csv"}
    )


# --- DM CSV 파일 다운로드 엔드포인트 ---
@app.get("/download_csv_dm", summary="DM CSV 파일 다운로드")
async def download_csv_dm_api():
    """
    마지막으로 계산된 DM 삽입 손실 및 S-파라미터 데이터를 CSV 파일 형식으로 반환합니다.
    """
    global last_calculated_s_params_dm, last_calculated_il_dm, last_calculated_freq

    if last_calculated_s_params_dm is None or last_calculated_il_dm is None or last_calculated_freq is None:
        raise HTTPException(status_code=404, detail="DM 데이터가 없습니다. 먼저 삽입 손실을 계산하세요.")

    freq = np.array(last_calculated_freq)
    il_dm_db = np.array(last_calculated_il_dm)
    s11_dm = np.array([complex(r, i) for r, i in last_calculated_s_params_dm["s11"]])
    s12_dm = np.array([complex(r, i) for r, i in last_calculated_s_params_dm["s12"]])
    s21_dm = np.array([complex(r, i) for r, i in last_calculated_s_params_dm["s21"]])
    s22_dm = np.array([complex(r, i) for r, i in last_calculated_s_params_dm["s22"]])

    df = pd.DataFrame({
        'Frequency (Hz)': freq,
        'DM_IL (dB)': il_dm_db,
        'S11_DM_Real': s11_dm.real,
        'S11_DM_Imag': s11_dm.imag,
        'S12_DM_Real': s12_dm.real,
        'S12_DM_Imag': s12_dm.imag,
        'S21_DM_Real': s21_dm.real,
        'S21_DM_Imag': s21_dm.imag,
        'S22_DM_Real': s22_dm.real,
        'S22_DM_Imag': s22_dm.imag
    })

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=filter_data_dm.csv"}
    )
# --- 서버 실행 ---
# 터미널에서 실행: uvicorn main_API:app --host 0.0.0.0 --port 8001 --reload
# 터미널에서 실행: uvicorn main_API:app --host 0.0.0.0 --port 9001 --reload