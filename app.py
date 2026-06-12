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
# App Config
# ======================================================

st.set_page_config(
    page_title="Water Quality Prediction with GEE + LLM",
    layout="wide"
)

DATA_FILE = "data/final_df_water_quality.csv"
MODEL_NAME = "google/flan-t5-small"

RISK_STYLE = {
    "Very Low": {"color": "#2E7D32", "emoji": "🟢", "intensity": 0.25},
    "Low": {"color": "#66BB6A", "emoji": "🟩", "intensity": 0.40},
    "Moderate": {"color": "#FB8C00", "emoji": "🟠", "intensity": 0.60},
    "High": {"color": "#E53935", "emoji": "🔴", "intensity": 0.75},
    "Very High": {"color": "#7F0000", "emoji": "🟥", "intensity": 0.90},
}


# ======================================================
# CSS
# ======================================================

st.markdown(
    """
    <style>
    .title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .subtitle {
        font-size: 16px;
        color: #666;
        margin-bottom: 18px;
    }
    .guide-box {
        background: #F3F7FF;
        border: 1px solid #D7E3FF;
        padding: 15px;
        border-radius: 14px;
        margin-bottom: 12px;
    }
    .risk-card {
        padding: 24px;
        border-radius: 18px;
        color: white;
        margin-bottom: 16px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.18);
    }
    .risk-big {
        font-size: 34px;
        font-weight: 800;
    }
    .risk-small {
        font-size: 16px;
        margin-top: 5px;
    }
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .legend-card {
        padding: 12px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-weight: 700;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Water Quality Prediction with Google Earth Engine + LLM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Select a map point or draw an area → extract latitude, longitude, altitude, and climate data → enter water-quality values → get prediction, map color, and chatbot explanation.</div>',
    unsafe_allow_html=True
)


# ======================================================
# Load Data
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
# Earth Engine
# ======================================================

@st.cache_resource
def init_earth_engine():
    if ee is None:
        return False, "earthengine-api is not installed."

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
            return True, "Google Earth Engine connected using Streamlit secrets."

        ee.Initialize()
        return True, "Google Earth Engine connected using local credentials."

    except Exception as e:
        return False, str(e)


gee_ready, gee_message = init_earth_engine()


# ======================================================
# LLM
# ======================================================

@st.cache_resource
def load_llm():
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None or torch is None:
        return None, None, "Transformers or torch is unavailable."

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.eval()
        return tokenizer, model, "LLM loaded."
    except Exception as e:
        return None, None, str(e)


tokenizer, llm_model, llm_message = load_llm()


# ======================================================
# Helper Functions
# ======================================================

def calculate_area_center_from_geometry(geometry):
    if not geometry:
        return None, None, "Point"

    geom_type = geometry.get("type")

    if geom_type == "Point":
        coords = geometry.get("coordinates", [])
        if len(coords) == 2:
            return coords[1], coords[0], "Point"

    if geom_type == "Polygon":
        coords = geometry.get("coordinates", [])
        try:
            points = coords[0]
            lons = [p[0] for p in points]
            lats = [p[1] for p in points]
            return sum(lats) / len(lats), sum(lons) / len(lons), "Polygon area"
        except Exception:
            pass

    return None, None, geom_type or "Selected area"


def extract_earth_engine_data(latitude, longitude):
    if not gee_ready:
        return {
            "altitude_m": "NA",
            "temperature_2m_C": "NA",
            "dewpoint_temperature_2m_C": "NA",
            "total_precipitation_sum": "NA",
            "surface_pressure": "NA",
            "source": "Earth Engine not initialized"
        }

    try:
        point = ee.Geometry.Point([longitude, latitude])

        climate = (
            ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
            .filterDate("2023-01-01", "2023-02-01")
            .first()
        )

        climate_values = climate.select([
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

        elevation = ee.Image("USGS/SRTMGL1_003").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(90),
            scale=30,
            bestEffort=True
        ).getInfo()

        temp_k = climate_values.get("temperature_2m")
        dew_k = climate_values.get("dewpoint_temperature_2m")

        return {
            "altitude_m": round(elevation.get("elevation"), 2) if elevation.get("elevation") is not None else "NA",
            "temperature_2m_C": round(temp_k - 273.15, 2) if temp_k is not None else "NA",
            "dewpoint_temperature_2m_C": round(dew_k - 273.15, 2) if dew_k is not None else "NA",
            "total_precipitation_sum": climate_values.get("total_precipitation_sum", "NA"),
            "surface_pressure": climate_values.get("surface_pressure", "NA"),
            "source": "ERA5-Land Monthly + SRTM elevation"
        }

    except Exception as e:
        return {
            "altitude_m": "NA",
            "temperature_2m_C": "NA",
            "dewpoint_temperature_2m_C": "NA",
            "total_precipitation_sum": "NA",
            "surface_pressure": "NA",
            "source": "Earth Engine extraction failed",
            "error": str(e)
        }


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
    elif score <= 3:
        level = "Low"
    elif score <= 5:
        level = "Moderate"
    elif score <= 7:
        level = "High"
    else:
        level = "Very High"

    if not reasons:
        reasons.append("all major parameters are within low-risk range")

    return (
        level,
        score,
        RISK_STYLE[level]["color"],
        RISK_STYLE[level]["intensity"],
        "; ".join(reasons)
    )


def parameter_status(param, value):
    value = float(value)

    if param == "pH":
        if 6.5 <= value <= 8.5:
            return "Good", "Within recommended range"
        return "Concern", "pH outside 6.5–8.5"

    if param == "TDS":
        if value <= 500:
            return "Good", "Low dissolved solids"
        if value <= 1000:
            return "Moderate", "Elevated dissolved solids"
        return "High", "Very high dissolved solids"

    if param == "TH":
        if value <= 300:
            return "Good", "Acceptable hardness"
        return "Concern", "Hard water"

    if param == "DO":
        if value >= 6:
            return "Good", "Healthy dissolved oxygen"
        if value >= 5:
            return "Moderate", "Borderline dissolved oxygen"
        return "Concern", "Low dissolved oxygen"

    if param == "BOD":
        if value <= 3:
            return "Good", "Low organic load"
        if value <= 6:
            return "Moderate", "Possible organic pollution"
        return "High", "High organic pollution"

    if param in ["Ca", "Mg"]:
        return "Info", "Mineral component"

    return "Info", "Recorded"


def find_nearest_dataframe_record(latitude, longitude):
    df = final_df.copy()
    df["distance"] = (
        (df["latitude"] - latitude) ** 2 +
        (df["longitude"] - longitude) ** 2
    ) ** 0.5
    return df.sort_values("distance").iloc[0]


def run_llm(prompt):
    if tokenizer is None or llm_model is None or torch is None:
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
                max_new_tokens=220,
                do_sample=False
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if len(text) < 30:
            return ""

        return text

    except Exception:
        return ""


def build_ai_context(input_data, earth_data, nearest_record, risk_level, risk_score, concerns):
    return f"""
Selected map area:
Latitude: {input_data['latitude']}
Longitude: {input_data['longitude']}
Altitude: {earth_data.get('altitude_m')} m

Google Earth Engine:
Temperature: {earth_data.get('temperature_2m_C')} C
Dewpoint: {earth_data.get('dewpoint_temperature_2m_C')} C
Precipitation: {earth_data.get('total_precipitation_sum')}
Surface pressure: {earth_data.get('surface_pressure')}

Water quality:
pH: {input_data['pH']}
TDS: {input_data['TDS']} mg/L
TH: {input_data['TH']} mg/L
Ca: {input_data['Ca']} mg/L
Mg: {input_data['Mg']} mg/L
DO: {input_data['DO']} mg/L
BOD: {input_data['BOD']} mg/L

Nearest dataframe reference:
City: {nearest_record.get('city', 'Unknown')}
Reference pH: {nearest_record.get('pH', 'NA')}
Reference TDS: {nearest_record.get('TDS', 'NA')}
Reference TH: {nearest_record.get('TH', 'NA')}
Reference Ca: {nearest_record.get('Ca', 'NA')}
Reference Mg: {nearest_record.get('Mg', 'NA')}
Reference DO: {nearest_record.get('DO', 'NA')}
Reference BOD: {nearest_record.get('BOD', 'NA')}

Prediction:
Risk level: {risk_level}
Risk score: {risk_score}
Concern parameters: {concerns}
"""


def generate_explanation(input_data, earth_data, nearest_record, risk_level, risk_score, color, intensity, concerns):
    context = build_ai_context(input_data, earth_data, nearest_record, risk_level, risk_score, concerns)

    prompt = f"""
You are a water quality expert. Explain the following result in simple scientific language.
{context}

Give:
overall status, concerning parameters, climate/altitude influence, possible pollution sources, and monitoring action.
"""

    llm_text = run_llm(prompt)

    if not llm_text:
        llm_text = (
            f"The selected area is classified as {risk_level} risk with score {risk_score}. "
            f"The main concern parameters are {concerns}. "
            "TDS and hardness reflect dissolved mineral loading, while DO and BOD reflect oxygen balance and possible organic pollution. "
            "Altitude and climate conditions may influence runoff, evaporation, dilution, and oxygen availability."
        )

    pollution_sources = []

    if input_data["BOD"] > 3:
        pollution_sources.append("organic pollution, sewage input, or decomposing biological matter")
    if input_data["DO"] < 5:
        pollution_sources.append("oxygen depletion, stagnant water, or microbial activity")
    if input_data["TDS"] > 500:
        pollution_sources.append("mineral loading, salinity, or dissolved solids")
    if input_data["TH"] > 300:
        pollution_sources.append("hardness from calcium/magnesium-rich water or groundwater influence")
    if input_data["pH"] < 6.5 or input_data["pH"] > 8.5:
        pollution_sources.append("acidic or alkaline stress")

    if not pollution_sources:
        pollution_sources.append("no strong pollution signal from the entered parameters")

    pollution_html = "".join([f"<li>{x}</li>" for x in pollution_sources])

    return f"""
<div class="risk-card" style="background:{color};">
    <div class="risk-big">{RISK_STYLE[risk_level]['emoji']} {risk_level} Risk</div>
    <div class="risk-small">Score: {risk_score} | Color Intensity: {intensity}</div>
    <div class="risk-small">Main concern: {concerns}</div>
</div>

### AI / LLM Interpretation

{llm_text}

### What the Result Means

The selected map area is predicted as **{risk_level} risk**.  
The prediction uses user-entered water chemistry, selected map coordinates, altitude, Earth Engine climate data, and the nearest reference record from your dataframe.

### Possible Pollution or Environmental Sources

<ul>
{pollution_html}
</ul>

### Recommended Monitoring

- Repeat sampling at the selected point or inside the drawn area.
- Compare current values with the nearest dataframe reference.
- If risk is Moderate, High, or Very High, inspect nearby drains, sewage discharge, industrial discharge, agricultural runoff, or stagnant water bodies.
- Continue monitoring pH, TDS, TH, Ca, Mg, DO, and BOD.
"""


def answer_chatbot(question):
    if not st.session_state.get("prediction_done"):
        return "Please first select a map location, enter water-quality values, and click **Predict and Explain**. Then I can answer questions about that selected area."

    input_data = st.session_state.input_data
    earth_data = st.session_state.earth_data
    nearest_record = st.session_state.nearest_record

    context = build_ai_context(
        input_data,
        earth_data,
        nearest_record,
        st.session_state.risk_level,
        st.session_state.risk_score,
        st.session_state.concerns
    )

    prompt = f"""
You are a water-quality chatbot. Answer the user's question using only this selected map area context.

{context}

User question:
{question}

Answer clearly and practically.
"""

    llm_answer = run_llm(prompt)

    if llm_answer:
        return llm_answer

    q = question.lower()

    if "bod" in q:
        return f"BOD is {input_data['BOD']} mg/L. Higher BOD means more organic matter and more oxygen demand. Concern status: {parameter_status('BOD', input_data['BOD'])[1]}."
    if "do" in q or "oxygen" in q:
        return f"DO is {input_data['DO']} mg/L. Low DO may indicate oxygen stress, stagnation, or organic pollution. Status: {parameter_status('DO', input_data['DO'])[1]}."
    if "tds" in q:
        return f"TDS is {input_data['TDS']} mg/L. High TDS may indicate dissolved minerals, salinity, or contamination input. Status: {parameter_status('TDS', input_data['TDS'])[1]}."
    if "altitude" in q or "elevation" in q:
        return f"The selected area's altitude from Earth Engine/SRTM is {earth_data.get('altitude_m')} meters."
    if "risk" in q:
        return f"The selected area is classified as {st.session_state.risk_level} risk with score {st.session_state.risk_score}. Main concerns: {st.session_state.concerns}."

    return (
        f"For the selected area, the risk level is {st.session_state.risk_level} with score {st.session_state.risk_score}. "
        f"Main concerns are {st.session_state.concerns}. Earth Engine altitude is {earth_data.get('altitude_m')} m."
    )


def add_map_legend(folium_map):
    legend_html = """
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        width: 185px;
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
# Session State
# ======================================================

defaults = {
    "latitude": float(final_df["latitude"].iloc[0]),
    "longitude": float(final_df["longitude"].iloc[0]),
    "selected_type": "Point",
    "drawn_area": None,
    "prediction_done": False,
    "chat_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ======================================================
# Sidebar
# ======================================================

with st.sidebar:
    st.header("How to Use")

    st.markdown(
        """
        1. Click on the map **or draw an area**.  
        2. The selected latitude/longitude will update.  
        3. Enter pH, TDS, TH, Ca, Mg, DO, and BOD.  
        4. Click **Predict and Explain**.  
        5. Ask follow-up questions in the chatbot.
        """
    )

    st.divider()

    st.header("System Status")

    if gee_ready:
        st.success("Google Earth Engine connected")
    else:
        st.warning("Earth Engine not connected")
        st.caption("Altitude and climate values will show NA.")

    if tokenizer is not None and llm_model is not None:
        st.success("LLM loaded")
    else:
        st.warning("LLM unavailable")
        st.caption("Structured fallback explanation will be used.")

    st.divider()

    st.header("Legend")
    for level, info in RISK_STYLE.items():
        st.markdown(
            f"""
            <div class="legend-card" style="background:{info['color']};">
            {info['emoji']} {level}
            </div>
            """,
            unsafe_allow_html=True
        )


# ======================================================
# Main Layout
# ======================================================

st.markdown(
    """
    <div class="guide-box">
    <b>Selected Area Workflow:</b> Use the map tools to select a point or draw a polygon/rectangle/circle.
    The app uses the selected center coordinate, latitude, longitude, altitude, Earth Engine climate data,
    your water-quality values, and your dataframe reference to generate prediction and explanation.
    </div>
    """,
    unsafe_allow_html=True
)

col_map, col_input = st.columns([2, 1])


# ======================================================
# Map Panel
# ======================================================

with col_map:
    st.subheader("1. Select Map Area")

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
        edit_options={"edit": True, "remove": True}
    ).add_to(m)

    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        popup=f"Selected {st.session_state.selected_type}",
        tooltip="Selected prediction center"
    ).add_to(m)

    if st.session_state.prediction_done:
        folium.CircleMarker(
            location=[st.session_state.latitude, st.session_state.longitude],
            radius=40,
            color=st.session_state.map_color,
            fill=True,
            fill_color=st.session_state.map_color,
            fill_opacity=st.session_state.intensity,
            popup=f"{st.session_state.risk_level} Risk | Score {st.session_state.risk_score}"
        ).add_to(m)

    add_map_legend(m)

    map_data = st_folium(
        m,
        height=650,
        width=950,
        returned_objects=["last_clicked", "last_active_drawing", "all_drawings"],
        key="water_quality_map"
    )

    if map_data:
        if map_data.get("last_clicked"):
            st.session_state.latitude = map_data["last_clicked"]["lat"]
            st.session_state.longitude = map_data["last_clicked"]["lng"]
            st.session_state.selected_type = "Clicked point"

        if map_data.get("last_active_drawing"):
            st.session_state.drawn_area = map_data["last_active_drawing"]
            geometry = st.session_state.drawn_area.get("geometry", {})
            lat, lon, selected_type = calculate_area_center_from_geometry(geometry)

            if lat is not None and lon is not None:
                st.session_state.latitude = lat
                st.session_state.longitude = lon
                st.session_state.selected_type = selected_type

    st.info(
        f"Selected for prediction: {st.session_state.selected_type} | "
        f"Latitude: {st.session_state.latitude:.6f}, Longitude: {st.session_state.longitude:.6f}"
    )


# ======================================================
# Input Panel
# ======================================================

with col_input:
    st.subheader("2. Confirm Location + Enter Water Values")

    with st.expander("Selected coordinate used for prediction", expanded=True):
        latitude = st.number_input("Latitude", value=float(st.session_state.latitude), format="%.6f")
        longitude = st.number_input("Longitude", value=float(st.session_state.longitude), format="%.6f")
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

    if st.button("Predict and Explain", use_container_width=True):
        earth_data = extract_earth_engine_data(latitude, longitude)
        nearest_record = find_nearest_dataframe_record(latitude, longitude)

        risk_level, risk_score, map_color, intensity, concerns = calculate_risk(input_data)

        explanation = generate_explanation(
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
        st.session_state.input_data = input_data
        st.session_state.earth_data = earth_data
        st.session_state.nearest_record = nearest_record
        st.session_state.risk_level = risk_level
        st.session_state.risk_score = risk_score
        st.session_state.map_color = map_color
        st.session_state.intensity = intensity
        st.session_state.concerns = concerns
        st.session_state.explanation = explanation


# ======================================================
# Parameter Dashboard
# ======================================================

st.divider()
st.subheader("3. Current Input Dashboard")

param_rows = []

for param in ["pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD"]:
    status, meaning = parameter_status(param, input_data[param])
    param_rows.append(
        {
            "Parameter": param,
            "Value": input_data[param],
            "Status": status,
            "Meaning": meaning
        }
    )

param_df = pd.DataFrame(param_rows)
st.dataframe(param_df, use_container_width=True)

st.bar_chart(
    pd.DataFrame({
        "Parameter": ["pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD"],
        "Value": [pH, TDS, TH, Ca, Mg, DO, BOD]
    }).set_index("Parameter")
)


# ======================================================
# Output
# ======================================================

if st.session_state.prediction_done:
    st.divider()
    st.subheader("4. Prediction Result")

    st.markdown(st.session_state.explanation, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Risk Level", st.session_state.risk_level)
    c2.metric("Risk Score", st.session_state.risk_score)
    c3.metric("Color", st.session_state.map_color)
    c4.metric("Intensity", st.session_state.intensity)
    c5.metric("Altitude", st.session_state.earth_data.get("altitude_m"))

    st.progress(min(st.session_state.risk_score / 9, 1.0))

    with st.expander("Google Earth Engine Data", expanded=True):
        st.json(st.session_state.earth_data)

    with st.expander("Nearest Dataframe Reference", expanded=True):
        nearest_df = pd.DataFrame([st.session_state.nearest_record])
        show_cols = [c for c in ["city", "latitude", "longitude", "pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD"] if c in nearest_df.columns]
        st.dataframe(nearest_df[show_cols], use_container_width=True)

    result_df = pd.DataFrame([{
        "latitude": st.session_state.latitude,
        "longitude": st.session_state.longitude,
        "selected_type": st.session_state.selected_type,
        "altitude_m": st.session_state.earth_data.get("altitude_m"),
        "Risk_Level": st.session_state.risk_level,
        "Risk_Score": st.session_state.risk_score,
        "Color": st.session_state.map_color,
        "Intensity": st.session_state.intensity,
        "Concern_Parameters": st.session_state.concerns,
        "pH": st.session_state.input_data["pH"],
        "TDS": st.session_state.input_data["TDS"],
        "TH": st.session_state.input_data["TH"],
        "Ca": st.session_state.input_data["Ca"],
        "Mg": st.session_state.input_data["Mg"],
        "DO": st.session_state.input_data["DO"],
        "BOD": st.session_state.input_data["BOD"],
    }])

    st.download_button(
        "Download Prediction CSV",
        result_df.to_csv(index=False),
        "water_quality_prediction_result.csv",
        "text/csv",
        use_container_width=True
    )

    if st.session_state.drawn_area is not None:
        with st.expander("Drawn Area GeoJSON"):
            st.json(st.session_state.drawn_area)


# ======================================================
# Chatbot
# ======================================================

st.divider()
st.subheader("5. Chatbot: Ask About the Selected Area")

question = st.text_input(
    "Ask a question",
    placeholder="Example: Why is this area high risk? What does BOD mean? How does altitude affect this water body?"
)

if st.button("Ask Chatbot", use_container_width=True):
    answer = answer_chatbot(question)
    st.session_state.chat_history.append({"question": question, "answer": answer})

for chat in reversed(st.session_state.chat_history[-5:]):
    st.markdown(f"**User:** {chat['question']}")
    st.markdown(f"**Assistant:** {chat['answer']}")
    st.divider()
