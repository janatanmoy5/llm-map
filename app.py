# app.py

import os
import json
import math
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

try:
    import ee
except Exception:
    ee = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None


# ======================================================
# APP CONFIG
# ======================================================

st.set_page_config(
    page_title="Water Quality Prediction with GEE + LLM",
    layout="wide"
)

DATA_FILE = "data/final_df_water_quality.csv"
MODEL_NAME = "google/flan-t5-small"

RISK_STYLE = {
    "Very Low": {
        "color": "#2E7D32",
        "emoji": "🟢",
        "label": "Very Low",
        "summary": "Generally acceptable water quality with minor concern."
    },
    "Low": {
        "color": "#66BB6A",
        "emoji": "🟩",
        "label": "Low",
        "summary": "Low risk, but continued monitoring is recommended."
    },
    "Moderate": {
        "color": "#FB8C00",
        "emoji": "🟠",
        "label": "Moderate",
        "summary": "Moderate concern; some parameters need attention."
    },
    "High": {
        "color": "#E53935",
        "emoji": "🔴",
        "label": "High",
        "summary": "High risk; pollution source investigation is recommended."
    },
    "Very High": {
        "color": "#7F0000",
        "emoji": "🟥",
        "label": "Very High",
        "summary": "Very high risk; urgent validation and action are recommended."
    }
}


# ======================================================
# CSS
# ======================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 16px;
        color: #666;
        margin-bottom: 20px;
    }
    .risk-card {
        padding: 22px;
        border-radius: 18px;
        color: white;
        margin-bottom: 16px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    }
    .risk-big {
        font-size: 34px;
        font-weight: 800;
    }
    .risk-small {
        font-size: 16px;
        margin-top: 6px;
    }
    .info-box {
        padding: 15px;
        border-radius: 12px;
        background: #F7F9FC;
        border: 1px solid #E0E5EF;
        margin-bottom: 12px;
    }
    .param-ok {
        color: #2E7D32;
        font-weight: 700;
    }
    .param-warn {
        color: #FB8C00;
        font-weight: 700;
    }
    .param-bad {
        color: #C62828;
        font-weight: 700;
    }
    .legend-box {
        padding: 12px;
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid #ddd;
        line-height: 1.8;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# HEADER
# ======================================================

st.markdown('<div class="main-title">Water Quality Prediction with Google Earth Engine + LLM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Map input + Earth Engine climate data + water-quality dataframe + AI interpretation</div>',
    unsafe_allow_html=True
)


# ======================================================
# LOAD DATA
# ======================================================

if not os.path.exists(DATA_FILE):
    st.error("Missing file: data/final_df_water_quality.csv")
    st.stop()

final_df = pd.read_csv(DATA_FILE)

required_cols = ["pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD"]

for col in required_cols:
    if col not in final_df.columns:
        st.error(f"Missing required column in CSV: {col}")
        st.stop()

if "city" not in final_df.columns:
    final_df["city"] = "Unknown"

if "latitude" not in final_df.columns:
    final_df["latitude"] = 22.5726

if "longitude" not in final_df.columns:
    final_df["longitude"] = 88.3639


# ======================================================
# EARTH ENGINE INIT
# ======================================================

@st.cache_resource
def init_earth_engine():
    if ee is None:
        return False, "earthengine-api package not available."

    try:
        if "gee_service_account" in st.secrets and "gee_private_key" in st.secrets:
            service_account = st.secrets["gee_service_account"]
            private_key_raw = st.secrets["gee_private_key"]

            if isinstance(private_key_raw, str):
                key_dict = json.loads(private_key_raw)
            else:
                key_dict = dict(private_key_raw)

            credentials = ee.ServiceAccountCredentials(
                service_account,
                key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
            return True, "Google Earth Engine connected with Streamlit secrets."

        ee.Initialize()
        return True, "Google Earth Engine connected with local credentials."

    except Exception as e:
        return False, str(e)


gee_ready, gee_message = init_earth_engine()


# ======================================================
# OPTIONAL LLM LOAD
# ======================================================

@st.cache_resource
def load_llm():
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None or torch is None:
        return None, None, "Transformers or torch is not available."

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.eval()
        return tokenizer, model, "LLM loaded successfully."
    except Exception as e:
        return None, None, str(e)


tokenizer, llm_model, llm_status = load_llm()


# ======================================================
# FUNCTIONS
# ======================================================

def calculate_risk(row):
    score = 0
    reasons = []

    pH = float(row["pH"])
    TDS = float(row["TDS"])
    TH = float(row["TH"])
    DO = float(row["DO"])
    BOD = float(row["BOD"])

    if pH < 6.5 or pH > 8.5:
        score += 2
        reasons.append("pH outside recommended range")

    if TDS > 500:
        score += 1
        reasons.append("high TDS")

    if TDS > 1000:
        score += 2
        reasons.append("very high TDS")

    if TH > 300:
        score += 1
        reasons.append("high total hardness")

    if DO < 5:
        score += 2
        reasons.append("low dissolved oxygen")

    if BOD > 3:
        score += 1
        reasons.append("high BOD")

    if BOD > 6:
        score += 2
        reasons.append("very high BOD")

    if score <= 1:
        level = "Very Low"
        intensity = 0.25
    elif score <= 3:
        level = "Low"
        intensity = 0.40
    elif score <= 5:
        level = "Moderate"
        intensity = 0.60
    elif score <= 7:
        level = "High"
        intensity = 0.75
    else:
        level = "Very High"
        intensity = 0.90

    if not reasons:
        reasons.append("all major parameters are within low-risk range")

    color = RISK_STYLE[level]["color"]

    return level, score, color, intensity, "; ".join(reasons)


def parameter_status(name, value):
    value = float(value)

    if name == "pH":
        if 6.5 <= value <= 8.5:
            return "Good", "param-ok", "Recommended range"
        return "Concern", "param-bad", "Outside 6.5–8.5"

    if name == "TDS":
        if value <= 500:
            return "Good", "param-ok", "Acceptable"
        if value <= 1000:
            return "Moderate", "param-warn", "Elevated"
        return "High", "param-bad", "Very high"

    if name == "TH":
        if value <= 300:
            return "Good", "param-ok", "Acceptable hardness"
        return "Concern", "param-warn", "Hard water"

    if name == "DO":
        if value >= 6:
            return "Good", "param-ok", "Healthy oxygen"
        if value >= 5:
            return "Moderate", "param-warn", "Borderline oxygen"
        return "Concern", "param-bad", "Low oxygen"

    if name == "BOD":
        if value <= 3:
            return "Good", "param-ok", "Low organic load"
        if value <= 6:
            return "Moderate", "param-warn", "Organic load concern"
        return "High", "param-bad", "High organic pollution"

    if name in ["Ca", "Mg"]:
        return "Info", "param-ok", "Mineral component"

    return "Info", "param-ok", "Recorded"


def extract_earth_engine_data(latitude, longitude):
    if not gee_ready:
        return {
            "temperature_2m_C": "NA",
            "dewpoint_temperature_2m_C": "NA",
            "total_precipitation_sum": "NA",
            "surface_pressure": "NA",
            "source": "Earth Engine not initialized"
        }

    try:
        point = ee.Geometry.Point([longitude, latitude])

        image = (
            ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
            .filterDate("2023-01-01", "2023-02-01")
            .first()
        )

        values = image.select([
            "temperature_2m",
            "dewpoint_temperature_2m",
            "total_precipitation_sum",
            "surface_pressure"
        ]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(1000),
            scale=1000,
            bestEffort=True
        ).getInfo()

        temp_k = values.get("temperature_2m")
        dew_k = values.get("dewpoint_temperature_2m")

        return {
            "temperature_2m_C": round(temp_k - 273.15, 2) if temp_k is not None else "NA",
            "dewpoint_temperature_2m_C": round(dew_k - 273.15, 2) if dew_k is not None else "NA",
            "total_precipitation_sum": values.get("total_precipitation_sum", "NA"),
            "surface_pressure": values.get("surface_pressure", "NA"),
            "source": "ECMWF ERA5-Land Monthly Aggregated"
        }

    except Exception as e:
        return {
            "temperature_2m_C": "NA",
            "dewpoint_temperature_2m_C": "NA",
            "total_precipitation_sum": "NA",
            "surface_pressure": "NA",
            "source": "Earth Engine extraction failed",
            "error": str(e)
        }


def find_nearest_dataframe_record(latitude, longitude):
    df = final_df.copy()
    df["distance"] = (
        (df["latitude"] - latitude) ** 2 +
        (df["longitude"] - longitude) ** 2
    ) ** 0.5
    return df.sort_values("distance").iloc[0]


def safe_llm_generate(prompt):
    if tokenizer is None or llm_model is None:
        return ""

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if len(text) < 30:
            return ""

        return text

    except Exception:
        return ""


def build_structured_interpretation(input_data, earth_data, nearest_record, risk_level, risk_score, color, intensity, concerns):
    pH = input_data["pH"]
    TDS = input_data["TDS"]
    TH = input_data["TH"]
    Ca = input_data["Ca"]
    Mg = input_data["Mg"]
    DO = input_data["DO"]
    BOD = input_data["BOD"]

    pollution_sources = []

    if BOD > 3:
        pollution_sources.append("organic pollution or sewage influence")
    if DO < 5:
        pollution_sources.append("oxygen depletion, stagnant water, or microbial activity")
    if TDS > 500:
        pollution_sources.append("mineral loading, salinity, or dissolved solids input")
    if TH > 300:
        pollution_sources.append("hardness from calcium/magnesium-rich water or groundwater influence")
    if pH < 6.5 or pH > 8.5:
        pollution_sources.append("acidic or alkaline stress")

    if not pollution_sources:
        pollution_sources.append("no strong pollution signal from the entered parameters")

    climate_line = "Earth Engine climate data are currently unavailable."
    if earth_data.get("temperature_2m_C") != "NA":
        climate_line = (
            f"The selected area has ERA5 temperature around {earth_data.get('temperature_2m_C')} °C, "
            f"dewpoint around {earth_data.get('dewpoint_temperature_2m_C')} °C, "
            f"and precipitation value {earth_data.get('total_precipitation_sum')}."
        )

    nearest_city = nearest_record.get("city", "Unknown")

    prompt = f"""
Explain water quality in simple terms.

Risk level: {risk_level}
Risk score: {risk_score}
pH: {pH}
TDS: {TDS}
TH: {TH}
Ca: {Ca}
Mg: {Mg}
DO: {DO}
BOD: {BOD}
Concern parameters: {concerns}
Earth Engine climate: {climate_line}
Nearest reference city: {nearest_city}

Write a concise scientific interpretation.
"""

    llm_generated = safe_llm_generate(prompt)

    if not llm_generated:
        llm_generated = (
            f"The selected area shows {risk_level.lower()} water-quality risk based on the entered chemistry values. "
            f"The main concern is: {concerns}. "
            f"The combination of DO, BOD, TDS, hardness, pH, calcium, and magnesium helps indicate whether the water may be affected by organic pollution, dissolved solids, or mineral loading."
        )

    interpretation = f"""
### {RISK_STYLE[risk_level]['emoji']} Overall Interpretation

The selected map area is classified as **{risk_level} risk** with a **risk score of {risk_score}**.  
The map color is shown as **{risk_level}** using the legend color below.

<div style="background:{color}; color:white; padding:14px; border-radius:12px; font-size:18px; font-weight:700;">
Risk Color: {risk_level} | Score: {risk_score} | Intensity: {intensity}
</div>

### Main Concern Parameters

**{concerns}**

### Water Chemistry Meaning

- **pH = {pH}**: indicates acidity or alkalinity.
- **TDS = {TDS} mg/L**: reflects dissolved salts and minerals.
- **TH = {TH} mg/L**: reflects total hardness.
- **Ca = {Ca} mg/L** and **Mg = {Mg} mg/L**: contribute to hardness and mineral content.
- **DO = {DO} mg/L**: low DO may indicate oxygen stress.
- **BOD = {BOD} mg/L**: high BOD may indicate organic pollution.

### Google Earth Engine Context

{climate_line}

### Nearest Dataframe Reference

The nearest reference record in your dataframe is **{nearest_city}**.  
This allows the app to compare the selected map location with previously stored water-quality observations.

### Possible Pollution / Environmental Sources

- {pollution_sources[0]}
"""
    for src in pollution_sources[1:]:
        interpretation += f"\n- {src}"

    interpretation += f"""

### LLM-Assisted Interpretation

{llm_generated}

### Recommended Action

- Repeat sampling from the selected point or drawn area.
- If risk is **Moderate, High, or Very High**, investigate nearby drains, sewage inputs, industrial discharge, agricultural runoff, or stagnant zones.
- Continue monitoring pH, TDS, TH, Ca, Mg, DO, and BOD.
"""

    return interpretation


def add_map_legend(folium_map):
    legend_html = """
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        width: 180px;
        z-index: 9999;
        background: white;
        border: 2px solid #999;
        border-radius: 10px;
        padding: 12px;
        font-size: 14px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.25);
    ">
    <b>Risk Legend</b><br>
    <span style="color:#2E7D32;">●</span> Very Low<br>
    <span style="color:#66BB6A;">●</span> Low<br>
    <span style="color:#FB8C00;">●</span> Moderate<br>
    <span style="color:#E53935;">●</span> High<br>
    <span style="color:#7F0000;">●</span> Very High
    </div>
    """
    folium_map.get_root().html.add_child(folium.Element(legend_html))


# ======================================================
# SESSION STATE
# ======================================================

defaults = {
    "latitude": float(final_df["latitude"].iloc[0]),
    "longitude": float(final_df["longitude"].iloc[0]),
    "drawn_area": None,
    "prediction_done": False,
    "risk_level": None,
    "risk_score": None,
    "map_color": None,
    "intensity": None,
    "concerns": None,
    "earth_data": None,
    "nearest_record": None,
    "llm_text": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:
    st.header("App Status")

    if gee_ready:
        st.success("Google Earth Engine connected")
    else:
        st.warning("Earth Engine not connected")
        st.caption("The app still works, but climate data will show NA.")

    if tokenizer is not None and llm_model is not None:
        st.success("LLM loaded")
    else:
        st.warning("LLM unavailable")
        st.caption("The app will use structured AI-style interpretation.")

    st.divider()

    st.header("Risk Legend")
    st.markdown(
        """
        <div class="legend-box">
        🟢 <b>Very Low</b><br>
        🟩 <b>Low</b><br>
        🟠 <b>Moderate</b><br>
        🔴 <b>High</b><br>
        🟥 <b>Very High</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.header("Data")
    st.write("Rows:", final_df.shape[0])
    st.write("Columns:", final_df.shape[1])


# ======================================================
# LAYOUT
# ======================================================

col_map, col_input = st.columns([2, 1])


# ======================================================
# MAP PANEL
# ======================================================

with col_map:
    st.subheader("1. Select Location on Map")

    m = folium.Map(
        location=[st.session_state.latitude, st.session_state.longitude],
        zoom_start=8,
        tiles="OpenStreetMap"
    )

    Draw(
        export=True,
        draw_options={
            "polyline": False,
            "rectangle": True,
            "polygon": True,
            "circle": True,
            "marker": True,
            "circlemarker": False
        },
        edit_options={"edit": True}
    ).add_to(m)

    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        popup="Selected Location",
        tooltip="Selected Location"
    ).add_to(m)

    if st.session_state.prediction_done:
        folium.CircleMarker(
            location=[st.session_state.latitude, st.session_state.longitude],
            radius=38,
            color=st.session_state.map_color,
            fill=True,
            fill_color=st.session_state.map_color,
            fill_opacity=st.session_state.intensity,
            popup=f"{st.session_state.risk_level} Risk | Score {st.session_state.risk_score}"
        ).add_to(m)

    add_map_legend(m)

    map_data = st_folium(
        m,
        height=620,
        width=900,
        returned_objects=["last_clicked", "last_active_drawing", "all_drawings"],
        key="water_quality_map"
    )

    if map_data:
        if map_data.get("last_clicked"):
            st.session_state.latitude = map_data["last_clicked"]["lat"]
            st.session_state.longitude = map_data["last_clicked"]["lng"]

        if map_data.get("last_active_drawing"):
            st.session_state.drawn_area = map_data["last_active_drawing"]
            geometry = st.session_state.drawn_area.get("geometry", {})
            geom_type = geometry.get("type")

            if geom_type == "Point":
                coords = geometry.get("coordinates", [])
                if len(coords) == 2:
                    st.session_state.longitude = coords[0]
                    st.session_state.latitude = coords[1]

            elif geom_type == "Polygon":
                coords = geometry.get("coordinates", [])
                try:
                    polygon_points = coords[0]
                    lons = [p[0] for p in polygon_points]
                    lats = [p[1] for p in polygon_points]
                    st.session_state.longitude = sum(lons) / len(lons)
                    st.session_state.latitude = sum(lats) / len(lats)
                except Exception:
                    pass


# ======================================================
# INPUT PANEL
# ======================================================

with col_input:
    st.subheader("2. Enter Water Quality Values")

    with st.expander("Selected Coordinates", expanded=True):
        latitude = st.number_input(
            "Latitude",
            value=float(st.session_state.latitude),
            format="%.6f"
        )

        longitude = st.number_input(
            "Longitude",
            value=float(st.session_state.longitude),
            format="%.6f"
        )

        st.session_state.latitude = latitude
        st.session_state.longitude = longitude

    pH = st.slider("pH", 0.0, 14.0, 7.2, 0.1)
    TDS = st.number_input("TDS mg/L", value=450.0, step=10.0)
    TH = st.number_input("Total Hardness mg/L", value=180.0, step=10.0)
    Ca = st.number_input("Calcium mg/L", value=60.0, step=1.0)
    Mg = st.number_input("Magnesium mg/L", value=25.0, step=1.0)
    DO = st.slider("Dissolved Oxygen mg/L", 0.0, 15.0, 6.5, 0.1)
    BOD = st.slider("BOD mg/L", 0.0, 20.0, 3.2, 0.1)

    input_data = {
        "latitude": latitude,
        "longitude": longitude,
        "pH": pH,
        "TDS": TDS,
        "TH": TH,
        "Ca": Ca,
        "Mg": Mg,
        "DO": DO,
        "BOD": BOD
    }

    preview_level, preview_score, preview_color, preview_intensity, preview_concerns = calculate_risk(input_data)

    st.markdown(
        f"""
        <div class="risk-card" style="background:{preview_color};">
            <div class="risk-big">{RISK_STYLE[preview_level]['emoji']} {preview_level}</div>
            <div class="risk-small">Preview score: {preview_score} | Intensity: {preview_intensity}</div>
            <div class="risk-small">{preview_concerns}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Predict and Explain", use_container_width=True):
        earth_data = extract_earth_engine_data(latitude, longitude)
        nearest_record = find_nearest_dataframe_record(latitude, longitude)

        risk_level, risk_score, map_color, intensity, concerns = calculate_risk(input_data)

        llm_text = build_structured_interpretation(
            input_data,
            earth_data,
            nearest_record,
            risk_level,
            risk_score,
            map_color,
            intensity,
            concerns
        )

        st.session_state.prediction_done = True
        st.session_state.earth_data = earth_data
        st.session_state.nearest_record = nearest_record
        st.session_state.risk_level = risk_level
        st.session_state.risk_score = risk_score
        st.session_state.map_color = map_color
        st.session_state.intensity = intensity
        st.session_state.concerns = concerns
        st.session_state.llm_text = llm_text


# ======================================================
# PARAMETER TABLE PREVIEW
# ======================================================

st.divider()
st.subheader("3. Parameter Status Dashboard")

param_rows = []

for param in ["pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD"]:
    status, css_class, note = parameter_status(param, input_data[param])
    param_rows.append(
        {
            "Parameter": param,
            "Value": input_data[param],
            "Status": status,
            "Interpretation": note
        }
    )

param_df = pd.DataFrame(param_rows)

st.dataframe(param_df, use_container_width=True)

chart_df = pd.DataFrame(
    {
        "Parameter": ["pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD"],
        "Value": [pH, TDS, TH, Ca, Mg, DO, BOD]
    }
)

st.bar_chart(chart_df.set_index("Parameter"))


# ======================================================
# OUTPUT
# ======================================================

if st.session_state.prediction_done:
    st.divider()
    st.subheader("4. Prediction Output")

    style = RISK_STYLE[st.session_state.risk_level]

    st.markdown(
        f"""
        <div class="risk-card" style="background:{st.session_state.map_color};">
            <div class="risk-big">{style['emoji']} {st.session_state.risk_level} Risk</div>
            <div class="risk-small">Score: {st.session_state.risk_score} | Color Intensity: {st.session_state.intensity}</div>
            <div class="risk-small">{style['summary']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Level", st.session_state.risk_level)
    c2.metric("Risk Score", st.session_state.risk_score)
    c3.metric("Legend Color", st.session_state.map_color)
    c4.metric("Intensity", st.session_state.intensity)

    st.progress(min(st.session_state.risk_score / 9, 1.0))

    st.markdown("### Color Legend Used in This Prediction")
    legend_cols = st.columns(5)

    for idx, level in enumerate(["Very Low", "Low", "Moderate", "High", "Very High"]):
        level_color = RISK_STYLE[level]["color"]
        legend_cols[idx].markdown(
            f"""
            <div style="background:{level_color}; color:white; padding:14px; border-radius:12px; text-align:center;">
            <b>{RISK_STYLE[level]['emoji']}<br>{level}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Concern Parameters")
    st.warning(st.session_state.concerns)

    cgee, cref = st.columns(2)

    with cgee:
        st.markdown("### Google Earth Engine Information")
        st.json(st.session_state.earth_data)

    with cref:
        st.markdown("### Nearest Dataframe Reference")
        nearest_df = pd.DataFrame([st.session_state.nearest_record])
        show_cols = [c for c in ["city", "latitude", "longitude", "pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD"] if c in nearest_df.columns]
        st.dataframe(nearest_df[show_cols], use_container_width=True)

    st.markdown("### LLM / AI Interpretation")
    st.markdown(st.session_state.llm_text, unsafe_allow_html=True)

    result_df = pd.DataFrame(
        [
            {
                "latitude": st.session_state.latitude,
                "longitude": st.session_state.longitude,
                "Risk_Level": st.session_state.risk_level,
                "Risk_Score": st.session_state.risk_score,
                "Color": st.session_state.map_color,
                "Intensity": st.session_state.intensity,
                "Concern_Parameters": st.session_state.concerns,
                "GEE_Temperature_C": st.session_state.earth_data.get("temperature_2m_C"),
                "GEE_Dewpoint_C": st.session_state.earth_data.get("dewpoint_temperature_2m_C"),
                "GEE_Precipitation": st.session_state.earth_data.get("total_precipitation_sum"),
                "GEE_Surface_Pressure": st.session_state.earth_data.get("surface_pressure")
            }
        ]
    )

    st.download_button(
        "Download Prediction Result CSV",
        result_df.to_csv(index=False),
        "water_quality_prediction_result.csv",
        "text/csv",
        use_container_width=True
    )

    if st.session_state.drawn_area is not None:
        with st.expander("Drawn Area GeoJSON"):
            st.json(st.session_state.drawn_area)
