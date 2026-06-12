# app.py

import os
import json
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

import ee
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ======================================================
# App setup
# ======================================================
st.set_page_config(
    page_title="Water Quality Prediction with GEE + LLM",
    layout="wide"
)

st.title("Water Quality Prediction with Google Earth Engine + LLM")
st.caption("Map input + Earth Engine climate data + water-quality dataframe + LLM explanation")


# ======================================================
# Paths
# ======================================================
DATA_FILE = "data/final_df_water_quality.csv"
MODEL_NAME = "google/flan-t5-small"


# ======================================================
# Load dataframe
# ======================================================
if not os.path.exists(DATA_FILE):
    st.error("Missing file: data/final_df_water_quality.csv")
    st.stop()

final_df = pd.read_csv(DATA_FILE)

required_cols = ["pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD"]

for col in required_cols:
    if col not in final_df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

if "city" not in final_df.columns:
    final_df["city"] = "Unknown"

if "latitude" not in final_df.columns:
    final_df["latitude"] = 22.5726

if "longitude" not in final_df.columns:
    final_df["longitude"] = 88.3639


# ======================================================
# Google Earth Engine initialization
# ======================================================
@st.cache_resource
def init_earth_engine():
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
            return True, "Google Earth Engine initialized using Streamlit secrets."

        else:
            ee.Initialize()
            return True, "Google Earth Engine initialized using local credentials."

    except Exception as e:
        return False, str(e)


gee_ready, gee_message = init_earth_engine()

if gee_ready:
    st.success(gee_message)
else:
    st.warning("Google Earth Engine is not initialized. App will run without Earth Engine data.")
    st.caption(gee_message)


# ======================================================
# Load LLM safely
# ======================================================
@st.cache_resource
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


tokenizer, llm_model = load_llm_model()


# ======================================================
# Earth Engine data extraction
# ======================================================
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


# ======================================================
# Risk calculation
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
        reasons.append("pH outside safe range")

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
        level, color, intensity = "Very Low", "green", 0.25
    elif score <= 3:
        level, color, intensity = "Low", "lightgreen", 0.40
    elif score <= 5:
        level, color, intensity = "Moderate", "orange", 0.60
    elif score <= 7:
        level, color, intensity = "High", "red", 0.75
    else:
        level, color, intensity = "Very High", "darkred", 0.90

    if not reasons:
        reasons.append("all major water-quality parameters are within low-risk range")

    return level, score, color, intensity, "; ".join(reasons)


# ======================================================
# Nearest dataframe record
# ======================================================
def find_nearest_dataframe_record(latitude, longitude):
    df = final_df.copy()

    df["distance"] = (
        (df["latitude"] - latitude) ** 2 +
        (df["longitude"] - longitude) ** 2
    ) ** 0.5

    return df.sort_values("distance").iloc[0]


# ======================================================
# LLM generation function
# ======================================================
def run_llm(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.3,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_llm_explanation(input_data, earth_data, nearest_record, risk_level, risk_score, color, intensity, concerns):
    prompt = f"""
You are a water quality and environmental monitoring expert.

Analyze the selected map area using Google Earth Engine climate data, user water-quality input, and historical dataframe reference data.

Selected location:
Latitude: {input_data['latitude']}
Longitude: {input_data['longitude']}

Google Earth Engine / ERA5 information:
Temperature: {earth_data.get('temperature_2m_C')} C
Dewpoint: {earth_data.get('dewpoint_temperature_2m_C')} C
Total precipitation: {earth_data.get('total_precipitation_sum')}
Surface pressure: {earth_data.get('surface_pressure')}

User water-quality input:
pH: {input_data['pH']}
TDS: {input_data['TDS']} mg/L
Total hardness: {input_data['TH']} mg/L
Calcium: {input_data['Ca']} mg/L
Magnesium: {input_data['Mg']} mg/L
Dissolved oxygen: {input_data['DO']} mg/L
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

Calculated risk:
Risk level: {risk_level}
Risk score: {risk_score}
Map color: {color}
Color intensity: {intensity}
Concern parameters: {concerns}

Give a concise explanation with:
1. Overall water quality status
2. Concerning parameters
3. How climate/map data may influence water quality
4. Possible pollution sources
5. Recommended monitoring action
"""

    llm_text = run_llm(prompt)

    final_text = f"""
### Water Quality LLM Explanation

**Risk Level:** {risk_level}  
**Risk Score:** {risk_score}  
**Map Color:** {color}  
**Color Intensity:** {intensity}  

**Concern Parameters:**  
{concerns}

**LLM Interpretation:**  
{llm_text}

**Scientific Summary:**  
The selected area is classified as **{risk_level}** risk. High TDS or total hardness may indicate mineral loading, salinity, or groundwater influence. Low DO and high BOD may indicate organic pollution, sewage contamination, stagnant water, or microbial oxygen demand.

**Recommended Action:**  
Repeat sampling from the selected point or drawn area. If risk is High or Very High, investigate nearby drains, industrial discharge, agricultural runoff, sewage discharge, or stagnant water zones.
"""

    return final_text


# ======================================================
# Session state
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
# Layout
# ======================================================
col1, col2 = st.columns([2, 1])


# ======================================================
# Map input
# ======================================================
with col1:
    st.subheader("Map Input")

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
        popup="Selected Point",
        tooltip="Selected Point"
    ).add_to(m)

    if st.session_state.prediction_done:
        folium.CircleMarker(
            location=[st.session_state.latitude, st.session_state.longitude],
            radius=35,
            color=st.session_state.map_color,
            fill=True,
            fill_color=st.session_state.map_color,
            fill_opacity=st.session_state.intensity,
            popup=f"{st.session_state.risk_level} Risk | Score {st.session_state.risk_score}"
        ).add_to(m)

    map_data = st_folium(
        m,
        height=620,
        width=900,
        returned_objects=[
            "last_clicked",
            "last_active_drawing",
            "all_drawings"
        ],
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
# User input
# ======================================================
with col2:
    st.subheader("Selected Location")

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

    st.subheader("Water Quality Input")

    pH = st.number_input("pH", value=7.2, step=0.1)
    TDS = st.number_input("TDS", value=450.0, step=10.0)
    TH = st.number_input("TH", value=180.0, step=10.0)
    Ca = st.number_input("Ca", value=60.0, step=1.0)
    Mg = st.number_input("Mg", value=25.0, step=1.0)
    DO = st.number_input("DO", value=6.5, step=0.1)
    BOD = st.number_input("BOD", value=3.2, step=0.1)

    st.subheader("Legend")
    st.markdown("""
    🟢 **Very Low**  
    🟩 **Low**  
    🟠 **Moderate**  
    🔴 **High**  
    🟥 **Very High**
    """)

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

        llm_text = generate_llm_explanation(
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
# Output
# ======================================================
if st.session_state.prediction_done:
    st.divider()

    st.subheader("Prediction Output")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Risk Level", st.session_state.risk_level)
    c2.metric("Risk Score", st.session_state.risk_score)
    c3.metric("Map Color", st.session_state.map_color)
    c4.metric("Intensity", st.session_state.intensity)

    st.write("**Concern Parameters:**", st.session_state.concerns)

    st.subheader("Google Earth Engine Information")
    st.json(st.session_state.earth_data)

    st.subheader("Nearest Dataframe Reference")
    st.dataframe(pd.DataFrame([st.session_state.nearest_record]))

    st.subheader("LLM Explanation")
    st.markdown(st.session_state.llm_text)

    if st.session_state.drawn_area is not None:
        with st.expander("Drawn Area GeoJSON"):
            st.json(st.session_state.drawn_area)
