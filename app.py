# app.py

import os
import json
import math
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    import ee
except Exception:
    ee = None


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Water Quality AI Map",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_FILE = "data/final_df_water_quality.csv"

RISK_STYLE = {
    "Very Low": {"color": "#2E7D32", "emoji": "🟢", "intensity": 0.25},
    "Low": {"color": "#66BB6A", "emoji": "🟩", "intensity": 0.40},
    "Moderate": {"color": "#FB8C00", "emoji": "🟠", "intensity": 0.60},
    "High": {"color": "#E53935", "emoji": "🔴", "intensity": 0.75},
    "Very High": {"color": "#7F0000", "emoji": "🟥", "intensity": 0.90},
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }
    .app-title {
        font-size: 34px;
        font-weight: 850;
        color: #0F172A;
        margin-bottom: 0px;
    }
    .app-subtitle {
        color: #64748B;
        font-size: 15px;
        margin-bottom: 16px;
    }
    .panel {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0px 5px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 14px;
    }
    .panel-title {
        font-size: 18px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 10px;
    }
    .hint-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E3A8A;
        border-radius: 14px;
        padding: 12px;
        font-size: 14px;
        margin-bottom: 12px;
    }
    .risk-card {
        padding: 22px;
        border-radius: 18px;
        color: white;
        box-shadow: 0px 8px 22px rgba(0,0,0,0.18);
        margin-bottom: 14px;
    }
    .risk-big {
        font-size: 31px;
        font-weight: 850;
        line-height: 1.1;
    }
    .risk-small {
        font-size: 14px;
        margin-top: 6px;
    }
    .legend-row {
        display: flex;
        align-items: center;
        margin-bottom: 7px;
        font-size: 14px;
        color: #334155;
    }
    .legend-dot {
        height: 13px;
        width: 13px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .chat-user {
        background: #E0F2FE;
        border-radius: 14px;
        padding: 10px 12px;
        margin-bottom: 8px;
        color: #075985;
        font-size: 14px;
    }
    .chat-agent {
        background: #F1F5F9;
        border-radius: 14px;
        padding: 10px 12px;
        margin-bottom: 14px;
        color: #334155;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">Water Quality AI Map</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Left: water inputs | Center: map circle selection | Right: chat agent | Bottom: prediction, ML, LLM context, and nearest city</div>',
    unsafe_allow_html=True,
)


# ============================================================
# LOAD CSV
# ============================================================

if not os.path.exists(DATA_FILE):
    st.error("Missing file: data/final_df_water_quality.csv")
    st.stop()

final_df = pd.read_csv(DATA_FILE)

if "Unnamed: 0" in final_df.columns:
    final_df = final_df.drop(columns=["Unnamed: 0"])

required_cols = [
    "city", "pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD",
    "Risk_Level", "latitude", "longitude"
]

for col in required_cols:
    if col not in final_df.columns:
        st.error(f"Missing required column in CSV: {col}")
        st.stop()

numeric_cols = [
    "temperature_2m", "temperature_2m_C",
    "dewpoint_temperature_2m", "dewpoint_temperature_2m_C",
    "total_precipitation_sum", "surface_pressure",
    "pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD",
    "Water_Quality_Risk_Score", "Risk_Score",
    "Color_Intensity", "latitude", "longitude"
]

for col in numeric_cols:
    if col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

final_df = final_df.dropna(subset=["latitude", "longitude"])


# ============================================================
# FULL DATAFRAME TO LLM CONTEXT
# ============================================================

def dataframe_to_llm_context(df):
    context = []

    for _, row in df.iterrows():
        city_text = f"""
City: {row.get('city', 'Unknown')}

Earth Engine Climate Data:
- Temperature: {row.get('temperature_2m_C', 'NA')} °C
- Dewpoint Temperature: {row.get('dewpoint_temperature_2m_C', 'NA')} °C
- Total Precipitation: {row.get('total_precipitation_sum', 'NA')}
- Surface Pressure: {row.get('surface_pressure', 'NA')}

Water Quality Parameters:
- pH: {row.get('pH', 'NA')}
- TDS: {row.get('TDS', 'NA')} mg/L
- Total Hardness: {row.get('TH', 'NA')} mg/L
- Calcium: {row.get('Ca', 'NA')} mg/L
- Magnesium: {row.get('Mg', 'NA')} mg/L
- Dissolved Oxygen: {row.get('DO', 'NA')} mg/L
- BOD: {row.get('BOD', 'NA')} mg/L

Prediction:
- Water Quality: {row.get('Water_Quality_Prediction', 'NA')}
- Risk Level: {row.get('Risk_Level', 'NA')}
- Risk Score: {row.get('Risk_Score', row.get('Water_Quality_Risk_Score', 'NA'))}
- Concern Parameters: {row.get('Concern_Parameters', 'NA')}
"""
        context.append(city_text)

    return "\n".join(context)


llm_context = dataframe_to_llm_context(final_df)


# ============================================================
# EARTH ENGINE
# ============================================================

@st.cache_resource
def init_earth_engine():
    if ee is None:
        return False, "earthengine-api is not installed."

    try:
        if "gee_service_account" in st.secrets and "gee_private_key" in st.secrets:
            service_account = st.secrets["gee_service_account"]
            private_key_raw = st.secrets["gee_private_key"]

            key_dict = (
                json.loads(private_key_raw)
                if isinstance(private_key_raw, str)
                else dict(private_key_raw)
            )

            credentials = ee.ServiceAccountCredentials(
                service_account,
                key_data=json.dumps(key_dict),
            )

            ee.Initialize(credentials)
            return True, "Google Earth Engine connected with Streamlit secrets."

        ee.Initialize()
        return True, "Google Earth Engine connected with local credentials."

    except Exception as e:
        return False, str(e)


gee_ready, gee_message = init_earth_engine()


def extract_earth_engine_data(latitude, longitude, radius_m):
    if not gee_ready:
        return {
            "altitude_m": "NA",
            "temperature_2m_C": "NA",
            "dewpoint_temperature_2m_C": "NA",
            "total_precipitation_sum": "NA",
            "surface_pressure": "NA",
            "source": "Earth Engine not initialized",
        }

    try:
        point = ee.Geometry.Point([longitude, latitude])
        region = point.buffer(radius_m)

        climate = (
            ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
            .filterDate("2023-01-01", "2023-02-01")
            .first()
        )

        climate_values = (
            climate.select([
                "temperature_2m",
                "dewpoint_temperature_2m",
                "total_precipitation_sum",
                "surface_pressure",
            ])
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=1000,
                bestEffort=True,
            )
            .getInfo()
        )

        elevation = (
            ee.Image("USGS/SRTMGL1_003")
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=30,
                bestEffort=True,
            )
            .getInfo()
        )

        temp_k = climate_values.get("temperature_2m")
        dew_k = climate_values.get("dewpoint_temperature_2m")

        return {
            "altitude_m": round(elevation.get("elevation"), 2)
            if elevation.get("elevation") is not None else "NA",
            "temperature_2m_C": round(temp_k - 273.15, 2)
            if temp_k is not None else "NA",
            "dewpoint_temperature_2m_C": round(dew_k - 273.15, 2)
            if dew_k is not None else "NA",
            "total_precipitation_sum": climate_values.get("total_precipitation_sum", "NA"),
            "surface_pressure": climate_values.get("surface_pressure", "NA"),
            "source": "ERA5-Land Monthly + SRTM elevation",
        }

    except Exception as e:
        return {
            "altitude_m": "NA",
            "temperature_2m_C": "NA",
            "dewpoint_temperature_2m_C": "NA",
            "total_precipitation_sum": "NA",
            "surface_pressure": "NA",
            "source": "Earth Engine extraction failed",
            "error": str(e),
        }


# ============================================================
# UTILITIES
# ============================================================

def circle_area_km2(radius_m):
    return math.pi * (radius_m / 1000) ** 2


def get_circle_from_geometry(drawing):
    if not drawing:
        return None, None, None

    geom = drawing.get("geometry", {})
    props = drawing.get("properties", {})

    geom_type = geom.get("type")
    coords = geom.get("coordinates", [])
    radius_m = props.get("radius")

    if geom_type == "Point" and len(coords) == 2:
        lon = coords[0]
        lat = coords[1]

        if radius_m is None:
            radius_m = 1000.0

        return float(lat), float(lon), float(radius_m)

    return None, None, None


def calculate_rule_risk(row):
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

    return level, score, "; ".join(reasons)


def parameter_status(param, value):
    value = float(value)

    if param == "pH":
        return ("Good", "Within recommended range") if 6.5 <= value <= 8.5 else ("Concern", "pH outside 6.5–8.5")

    if param == "TDS":
        if value <= 500:
            return "Good", "Low dissolved solids"
        if value <= 1000:
            return "Moderate", "Elevated dissolved solids"
        return "High", "Very high dissolved solids"

    if param == "TH":
        return ("Good", "Acceptable hardness") if value <= 300 else ("Concern", "Hard water")

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


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c


def find_nearest_dataframe_record(latitude, longitude):
    df = final_df.copy()

    df["distance_km"] = df.apply(
        lambda r: haversine_km(latitude, longitude, r["latitude"], r["longitude"]),
        axis=1,
    )

    return df.sort_values("distance_km").iloc[0]


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


# ============================================================
# ML MODEL
# ============================================================

ML_FEATURES = [
    "latitude",
    "longitude",
    "pH",
    "TDS",
    "TH",
    "Ca",
    "Mg",
    "DO",
    "BOD",
    "temperature_2m_C",
    "dewpoint_temperature_2m_C",
    "total_precipitation_sum",
    "surface_pressure",
]


@st.cache_resource
def train_ml_model(df):
    train_df = df.copy()

    for col in ML_FEATURES:
        if col not in train_df.columns:
            train_df[col] = 0

        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

    train_df[ML_FEATURES] = train_df[ML_FEATURES].fillna(0)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_df["Risk_Level"].astype(str))

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(train_df[ML_FEATURES], y)

    return model, label_encoder


ml_model, label_encoder = train_ml_model(final_df)


def predict_ml_risk(input_data, earth_data):
    row = {
        "latitude": input_data["latitude"],
        "longitude": input_data["longitude"],
        "pH": input_data["pH"],
        "TDS": input_data["TDS"],
        "TH": input_data["TH"],
        "Ca": input_data["Ca"],
        "Mg": input_data["Mg"],
        "DO": input_data["DO"],
        "BOD": input_data["BOD"],
        "temperature_2m_C": earth_data.get("temperature_2m_C"),
        "dewpoint_temperature_2m_C": earth_data.get("dewpoint_temperature_2m_C"),
        "total_precipitation_sum": earth_data.get("total_precipitation_sum"),
        "surface_pressure": earth_data.get("surface_pressure"),
    }

    pred_df = pd.DataFrame([row])

    for col in ML_FEATURES:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")

    pred_df[ML_FEATURES] = pred_df[ML_FEATURES].fillna(0)

    pred_code = ml_model.predict(pred_df[ML_FEATURES])[0]
    pred_level = label_encoder.inverse_transform([pred_code])[0]

    probabilities = ml_model.predict_proba(pred_df[ML_FEATURES])[0]

    prob_df = pd.DataFrame(
        {
            "Risk_Level": label_encoder.classes_,
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False)

    return pred_level, prob_df


# ============================================================
# LLM-STYLE EXPLANATION
# ============================================================

def build_llm_prompt(
    input_data,
    earth_data,
    nearest_record,
    ml_risk_level,
    rule_risk_level,
    rule_score,
    color,
    intensity,
    concerns,
):
    selected_context = f"""
Selected User Map Input:
- Latitude: {input_data['latitude']}
- Longitude: {input_data['longitude']}
- Circle Radius: {input_data['radius_m']} meters
- Circle Area: {input_data['area_km2']:.3f} km²

Google Earth Engine / Map-Derived Data:
- Altitude: {earth_data.get('altitude_m')} meters
- Temperature: {earth_data.get('temperature_2m_C')} °C
- Dewpoint Temperature: {earth_data.get('dewpoint_temperature_2m_C')} °C
- Total Precipitation: {earth_data.get('total_precipitation_sum')}
- Surface Pressure: {earth_data.get('surface_pressure')}
- Source: {earth_data.get('source')}

User Water Quality Input:
- pH: {input_data['pH']}
- TDS: {input_data['TDS']} mg/L
- Total Hardness: {input_data['TH']} mg/L
- Calcium: {input_data['Ca']} mg/L
- Magnesium: {input_data['Mg']} mg/L
- Dissolved Oxygen: {input_data['DO']} mg/L
- BOD: {input_data['BOD']} mg/L

Nearest Reference City:
- City: {nearest_record.get('city', 'Unknown')}
- Distance: {nearest_record.get('distance_km', 0):.2f} km
- Reference pH: {nearest_record.get('pH', 'NA')}
- Reference TDS: {nearest_record.get('TDS', 'NA')}
- Reference TH: {nearest_record.get('TH', 'NA')}
- Reference Ca: {nearest_record.get('Ca', 'NA')}
- Reference Mg: {nearest_record.get('Mg', 'NA')}
- Reference DO: {nearest_record.get('DO', 'NA')}
- Reference BOD: {nearest_record.get('BOD', 'NA')}

Model Prediction:
- ML Risk Level: {ml_risk_level}
- Rule-Based Risk Level: {rule_risk_level}
- Rule-Based Risk Score: {rule_score}
- Color: {color}
- Color Intensity: {intensity}
- Concern Parameters: {concerns}
"""

    return f"""
You are a water quality and environmental monitoring expert.

Use the following training/reference dataset context:

{llm_context}

Now analyze this selected map area:

{selected_context}

Prepare a clear interpretation with:
1. Overall water quality assessment
2. Why the model predicted this risk level
3. Which parameters are most concerning
4. Comparison with nearest reference city
5. How altitude/climate may influence water quality
6. Possible pollution sources
7. Monitoring and management recommendations
"""


def generate_explanation(
    input_data,
    earth_data,
    nearest_record,
    ml_risk_level,
    rule_risk_level,
    rule_score,
    color,
    intensity,
    concerns,
    probability_df,
):
    llm_prompt = build_llm_prompt(
        input_data,
        earth_data,
        nearest_record,
        ml_risk_level,
        rule_risk_level,
        rule_score,
        color,
        intensity,
        concerns,
    )

    return f"""
<div class="risk-card" style="background:{color};">
    <div class="risk-big">{RISK_STYLE[ml_risk_level]['emoji']} {ml_risk_level} Risk</div>
    <div class="risk-small">ML prediction using map input + water-quality dataframe</div>
    <div class="risk-small">Rule score: {rule_score} | Color intensity: {intensity}</div>
    <div class="risk-small">Main concerns: {concerns}</div>
</div>

### LLM Model Input Context

The app builds an LLM-style prompt from full dataframe context, selected map circle, user water-quality inputs, Earth Engine values, ML prediction, and nearest reference city.

```text
{llm_prompt}
```

### LLM-Style Interpretation

The selected circle area is predicted as **{ml_risk_level} risk**.

The nearest reference city is **{nearest_record.get('city', 'Unknown')}**, located approximately **{nearest_record.get('distance_km', 0):.2f} km** from the selected circle center.

### Why This Risk Level Was Predicted

The main concern parameters are:

**{concerns}**

The ML model was trained from `final_df_water_quality.csv`, where each city has water-quality parameters, climate features, and a labeled risk level. The selected area is compared against those learned patterns.

### Parameter Interpretation

- **pH** indicates acidity or alkalinity.
- **TDS** indicates dissolved salts, minerals, or contamination input.
- **TH, Ca, and Mg** indicate hardness and mineral load.
- **DO** indicates oxygen availability for aquatic life.
- **BOD** indicates organic pollution and oxygen demand.

### Recommendation

Repeat sampling within the selected circle area. If risk is **Moderate, High, or Very High**, inspect sewage discharge, drains, industrial discharge, agricultural runoff, stagnant water, and possible high-mineral groundwater sources.
"""


# ============================================================
# CHAT AGENT
# ============================================================

def answer_chatbot(question):
    if not st.session_state.get("prediction_done"):
        return "Please draw a circle, enter water-quality values, and click **Search Water Quality** first."

    q = question.lower()
    input_data = st.session_state.input_data
    earth_data = st.session_state.earth_data
    nearest = st.session_state.nearest_record

    if "llm" in q or "context" in q or "prompt" in q:
        return (
            "The app prepares an LLM-style context from the full dataframe, selected map coordinates, "
            "circle radius, Earth Engine variables, water-quality inputs, ML prediction, risk score, "
            "and nearest reference city."
        )

    if "nearest" in q or "city" in q or "reference" in q:
        return (
            f"The nearest reference city is {nearest.get('city', 'Unknown')}, "
            f"about {nearest.get('distance_km', 0):.2f} km from the selected circle center."
        )

    if "risk" in q or "prediction" in q or "result" in q:
        return (
            f"The ML model predicts {st.session_state.ml_risk_level} risk. "
            f"The rule-based score is {st.session_state.rule_score}. "
            f"Main concerns: {st.session_state.concerns}."
        )

    if "bod" in q:
        return f"BOD is {input_data['BOD']} mg/L. Higher BOD suggests organic pollution and oxygen demand."

    if "do" in q or "oxygen" in q:
        return f"DO is {input_data['DO']} mg/L. Low DO may indicate oxygen stress, stagnation, or sewage influence."

    if "tds" in q:
        return f"TDS is {input_data['TDS']} mg/L. High TDS can indicate mineral loading, salinity, or contamination input."

    if "ph" in q:
        return f"pH is {input_data['pH']}. Recommended range is usually 6.5–8.5."

    if "hardness" in q or "th" in q:
        return f"Total hardness is {input_data['TH']} mg/L. High hardness is usually related to calcium and magnesium-rich water."

    if "altitude" in q or "elevation" in q:
        return f"Altitude for the selected circle area is {earth_data.get('altitude_m')} meters."

    if "area" in q or "radius" in q or "circle" in q:
        return f"The selected circle radius is {input_data['radius_m']} meters and area is {input_data['area_km2']:.3f} km²."

    return (
        f"The selected circle area is predicted as {st.session_state.ml_risk_level} risk. "
        f"The nearest reference city is {nearest.get('city', 'Unknown')}. "
        f"Main concerns: {st.session_state.concerns}."
    )


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "latitude": float(final_df["latitude"].iloc[0]),
    "longitude": float(final_df["longitude"].iloc[0]),
    "radius_m": 1000.0,
    "drawn_area": None,
    "prediction_done": False,
    "chat_history": [],
    "last_input_signature": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# THREE PANEL GUI
# ============================================================

left_col, map_col, chat_col = st.columns([1.05, 2.15, 1.15])


# ============================================================
# LEFT PANEL
# ============================================================

with left_col:
    st.markdown('<div class="panel"><div class="panel-title">1. Water Inputs</div>', unsafe_allow_html=True)

    with st.expander("Circle area", expanded=True):
        latitude = st.number_input(
            "Center latitude",
            value=float(st.session_state.latitude),
            format="%.6f",
        )

        longitude = st.number_input(
            "Center longitude",
            value=float(st.session_state.longitude),
            format="%.6f",
        )

        radius_m = st.number_input(
            "Radius meters",
            value=float(st.session_state.radius_m),
            min_value=50.0,
            step=100.0,
        )

        st.session_state.latitude = latitude
        st.session_state.longitude = longitude
        st.session_state.radius_m = radius_m

        st.caption(f"Area: {circle_area_km2(radius_m):.3f} km²")

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
        "radius_m": radius_m,
        "area_km2": circle_area_km2(radius_m),
        "pH": pH,
        "TDS": TDS,
        "TH": TH,
        "Ca": Ca,
        "Mg": Mg,
        "DO": DO,
        "BOD": BOD,
    }

    current_signature = tuple(input_data.values())

    if st.session_state.last_input_signature is None:
        st.session_state.last_input_signature = current_signature

    if current_signature != st.session_state.last_input_signature:
        st.session_state.prediction_done = False
        st.session_state.last_input_signature = current_signature

    search_clicked = st.button("Search Water Quality", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Legend</div>', unsafe_allow_html=True)

    for level, info in RISK_STYLE.items():
        st.markdown(
            f"""
            <div class="legend-row">
                <span class="legend-dot" style="background:{info['color']};"></span>
                {info['emoji']} {level}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAP PANEL
# ============================================================

with map_col:
    st.markdown('<div class="panel"><div class="panel-title">2. Circle Map Area</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hint-box">
        Draw or edit one circle. The app uses the circle center and radius for prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    m = folium.Map(
        location=[st.session_state.latitude, st.session_state.longitude],
        zoom_start=6,
        tiles="OpenStreetMap",
    )

    Draw(
        export=True,
        draw_options={
            "polyline": False,
            "rectangle": False,
            "polygon": False,
            "circle": True,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    folium.Circle(
        location=[st.session_state.latitude, st.session_state.longitude],
        radius=st.session_state.radius_m,
        color="#2563EB",
        fill=True,
        fill_color="#93C5FD",
        fill_opacity=0.22,
        popup="Selected circle area",
    ).add_to(m)

    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        popup="Circle center used for prediction",
        tooltip="Circle center",
    ).add_to(m)

    if st.session_state.prediction_done:
        folium.Circle(
            location=[st.session_state.latitude, st.session_state.longitude],
            radius=st.session_state.radius_m,
            color=st.session_state.map_color,
            fill=True,
            fill_color=st.session_state.map_color,
            fill_opacity=st.session_state.intensity,
            popup=f"{st.session_state.ml_risk_level} Risk",
        ).add_to(m)

    add_map_legend(m)

    map_data = st_folium(
        m,
        height=610,
        width=900,
        returned_objects=["last_active_drawing", "all_drawings"],
        key="water_quality_circle_map",
    )

    if map_data and map_data.get("last_active_drawing"):
        st.session_state.drawn_area = map_data["last_active_drawing"]

        lat, lon, drawn_radius = get_circle_from_geometry(st.session_state.drawn_area)

        if lat is not None and lon is not None:
            st.session_state.latitude = lat
            st.session_state.longitude = lon
            st.session_state.radius_m = drawn_radius
            st.session_state.prediction_done = False

    st.info(
        f"Circle center: {st.session_state.latitude:.6f}, {st.session_state.longitude:.6f} | "
        f"Radius: {st.session_state.radius_m:.1f} m | "
        f"Area: {circle_area_km2(st.session_state.radius_m):.3f} km²"
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# CHAT PANEL
# ============================================================

with chat_col:
    st.markdown('<div class="panel"><div class="panel-title">3. Chat Agent</div>', unsafe_allow_html=True)

    st.caption("Ask questions after Search Water Quality.")

    question = st.text_area(
        "Ask about selected area",
        placeholder="Example: Why is this risk level predicted?",
        height=110,
    )

    ask_clicked = st.button("Ask Chat Agent", use_container_width=True)

    if ask_clicked and question.strip():
        answer = answer_chatbot(question)
        st.session_state.chat_history.append({"question": question, "answer": answer})

    for chat in reversed(st.session_state.chat_history[-5:]):
        st.markdown(f'<div class="chat-user"><b>User:</b> {chat["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-agent"><b>Agent:</b> {chat["answer"]}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">System</div>', unsafe_allow_html=True)

    if gee_ready:
        st.success("Earth Engine connected")
    else:
        st.warning("Earth Engine not connected")
        st.caption("Climate and altitude will show NA.")

    st.write("Training rows:", len(final_df))
    st.write("ML model: Random Forest")
    st.write("LLM context: Prepared from dataframe")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SEARCH ACTION
# ============================================================

if search_clicked:
    earth_data = extract_earth_engine_data(latitude, longitude, radius_m)
    nearest_record = find_nearest_dataframe_record(latitude, longitude)

    rule_level, rule_score, concerns = calculate_rule_risk(input_data)
    ml_risk_level, probability_df = predict_ml_risk(input_data, earth_data)

    if ml_risk_level not in RISK_STYLE:
        ml_risk_level = rule_level

    map_color = RISK_STYLE[ml_risk_level]["color"]
    intensity = RISK_STYLE[ml_risk_level]["intensity"]

    explanation = generate_explanation(
        input_data,
        earth_data,
        nearest_record,
        ml_risk_level,
        rule_level,
        rule_score,
        map_color,
        intensity,
        concerns,
        probability_df,
    )

    st.session_state.prediction_done = True
    st.session_state.input_data = input_data
    st.session_state.earth_data = earth_data
    st.session_state.nearest_record = nearest_record
    st.session_state.ml_risk_level = ml_risk_level
    st.session_state.rule_risk_level = rule_level
    st.session_state.rule_score = rule_score
    st.session_state.map_color = map_color
    st.session_state.intensity = intensity
    st.session_state.concerns = concerns
    st.session_state.explanation = explanation
    st.session_state.probability_df = probability_df


# ============================================================
# DASHBOARD + RESULT
# ============================================================

st.divider()

dash_col, result_col = st.columns([1.1, 2.1])

with dash_col:
    st.markdown('<div class="panel"><div class="panel-title">Input Dashboard</div>', unsafe_allow_html=True)

    param_rows = []

    for param in ["pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD"]:
        status, meaning = parameter_status(param, input_data[param])
        param_rows.append(
            {
                "Parameter": param,
                "Value": input_data[param],
                "Status": status,
                "Meaning": meaning,
            }
        )

    st.dataframe(pd.DataFrame(param_rows), use_container_width=True, height=280)

    chart_df = pd.DataFrame(
        {
            "Parameter": ["pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD"],
            "Value": [pH, TDS, TH, Ca, Mg, DO, BOD],
        }
    )

    st.bar_chart(chart_df.set_index("Parameter"))

    st.markdown("</div>", unsafe_allow_html=True)


with result_col:
    st.markdown('<div class="panel"><div class="panel-title">Search Result</div>', unsafe_allow_html=True)

    if not st.session_state.prediction_done:
        st.info("Draw a circle, enter inputs, then click Search Water Quality.")
    else:
        st.markdown(st.session_state.explanation, unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("ML Risk", st.session_state.ml_risk_level)
        c2.metric("Rule Score", st.session_state.rule_score)
        c3.metric("Color", st.session_state.map_color)
        c4.metric("Intensity", st.session_state.intensity)
        c5.metric("Nearest City", st.session_state.nearest_record.get("city", "NA"))

        st.progress(min(st.session_state.rule_score / 9, 1.0))

        with st.expander("ML Class Probabilities", expanded=False):
            st.dataframe(st.session_state.probability_df, use_container_width=True)
            st.bar_chart(st.session_state.probability_df.set_index("Risk_Level")["Probability"])

        with st.expander("Nearest Reference City", expanded=False):
            nearest_df = pd.DataFrame([st.session_state.nearest_record])
            show_cols = [
                c
                for c in [
                    "city", "distance_km", "latitude", "longitude",
                    "pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD",
                    "Risk_Level", "Risk_Score",
                ]
                if c in nearest_df.columns
            ]
            st.dataframe(nearest_df[show_cols], use_container_width=True)

        with st.expander("Google Earth Engine Data", expanded=False):
            st.json(st.session_state.earth_data)

        with st.expander("Prepared LLM Context from Full Dataframe", expanded=False):
            st.text(llm_context)

        result_df = pd.DataFrame(
            [
                {
                    "latitude": st.session_state.latitude,
                    "longitude": st.session_state.longitude,
                    "radius_m": st.session_state.radius_m,
                    "area_km2": circle_area_km2(st.session_state.radius_m),
                    "nearest_city": st.session_state.nearest_record.get("city", "NA"),
                    "nearest_distance_km": st.session_state.nearest_record.get("distance_km", "NA"),
                    "ML_Risk_Level": st.session_state.ml_risk_level,
                    "Rule_Risk_Level": st.session_state.rule_risk_level,
                    "Rule_Score": st.session_state.rule_score,
                    "Color": st.session_state.map_color,
                    "Intensity": st.session_state.intensity,
                    "Concern_Parameters": st.session_state.concerns,
                    "altitude_m": st.session_state.earth_data.get("altitude_m"),
                    "pH": st.session_state.input_data["pH"],
                    "TDS": st.session_state.input_data["TDS"],
                    "TH": st.session_state.input_data["TH"],
                    "Ca": st.session_state.input_data["Ca"],
                    "Mg": st.session_state.input_data["Mg"],
                    "DO": st.session_state.input_data["DO"],
                    "BOD": st.session_state.input_data["BOD"],
                }
            ]
        )

        st.download_button(
            "Download Search Result CSV",
            result_df.to_csv(index=False),
            "water_quality_search_result.csv",
            "text/csv",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
