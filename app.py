
# app.py
 
import os
import re
import math
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
 
# ---- Planetary Computer / STAC imports ----
try:
    import pystac_client
    import planetary_computer
    import rasterio
    from rasterio.enums import Resampling
    import stackstac
    STAC_READY = True
except Exception:
    STAC_READY = False
 
warnings.filterwarnings("ignore")
 
 
# ============================================================
# APP CONFIG
# ============================================================
 
st.set_page_config(
    page_title="Water Quality AI Map",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="💧",
)
 
DATA_FILE = "data/final_df_water_quality.csv"
 
RISK_STYLE = {
    "Very Low":  {"color": "#2E7D32", "emoji": "🟢", "intensity": 0.25},
    "Low":       {"color": "#66BB6A", "emoji": "🟩", "intensity": 0.40},
    "Moderate":  {"color": "#FB8C00", "emoji": "🟠", "intensity": 0.60},
    "High":      {"color": "#E53935", "emoji": "🔴", "intensity": 0.75},
    "Very High": {"color": "#7F0000", "emoji": "🟥", "intensity": 0.90},
}
 
PC_STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
 
 
# ============================================================
# CSS
# ============================================================
 
st.markdown("""
<style>
/* ── Reset & base ──────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
 
/* Remove Streamlit default top padding and make full-width */
.block-container {
    padding-top: 0 !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}
 
/* Hide default Streamlit header/footer chrome */
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
 
/* ── Sticky app header bar ─────────────────────────────────────── */
.app-header {
    position: sticky;
    top: 0;
    z-index: 9999;
    background: linear-gradient(90deg, #0F172A 0%, #1E3A5F 100%);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    margin-left: -0.75rem;
    margin-right: -0.75rem;
    margin-bottom: 12px;
    width: calc(100% + 1.5rem);
}
.app-header-left {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
}
.app-header-icon {
    font-size: 28px;
    flex-shrink: 0;
}
.app-title {
    font-size: clamp(16px, 2.5vw, 26px);
    font-weight: 800;
    color: #FFFFFF;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin: 0;
    line-height: 1.2;
}
.app-subtitle {
    font-size: clamp(10px, 1.3vw, 13px);
    color: #94A3B8;
    margin: 0;
    line-height: 1.3;
}
.app-header-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    flex-shrink: 0;
}
.badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: #E2E8F0;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
}
.badge-green  { background: rgba(34,197,94,0.2);  border-color: rgba(34,197,94,0.4);  color: #86EFAC; }
.badge-blue   { background: rgba(59,130,246,0.2);  border-color: rgba(59,130,246,0.4); color: #93C5FD; }
.badge-orange { background: rgba(251,146,60,0.2);  border-color: rgba(251,146,60,0.4); color: #FED7AA; }
 
/* ── Panels ────────────────────────────────────────────────────── */
.panel {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 2px 10px rgba(15,23,42,0.06);
    margin-bottom: 12px;
    height: auto;
    overflow: visible;
}
.panel-title {
    font-size: clamp(13px, 1.5vw, 16px);
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
    border-bottom: 2px solid #F1F5F9;
    padding-bottom: 8px;
}
 
/* ── Hint box ──────────────────────────────────────────────────── */
.hint-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1E3A8A;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: clamp(11px, 1.2vw, 13px);
    margin-bottom: 10px;
    line-height: 1.5;
}
 
/* ── Risk card ─────────────────────────────────────────────────── */
.risk-card {
    padding: 16px 18px;
    border-radius: 14px;
    color: white;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    margin-bottom: 12px;
}
.risk-big   { font-size: clamp(20px, 2.5vw, 28px); font-weight: 850; line-height: 1.2; }
.risk-small { font-size: clamp(11px, 1.2vw, 13px); margin-top: 5px; opacity: 0.92; }
 
/* ── Legend ────────────────────────────────────────────────────── */
.legend-row   { display:flex; align-items:center; margin-bottom:5px; font-size:clamp(11px,1.2vw,13px); color:#334155; }
.legend-dot   { height:11px; width:11px; border-radius:50%; display:inline-block; margin-right:7px; flex-shrink:0; }
 
/* ── Chat bubbles ──────────────────────────────────────────────── */
.chat-scroll {
    max-height: 340px;
    overflow-y: auto;
    padding-right: 4px;
    scroll-behavior: smooth;
}
.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
 
.chat-bubble-user {
    background: linear-gradient(135deg,#DBEAFE,#EFF6FF);
    border: 1px solid #BFDBFE;
    border-radius: 16px 16px 4px 16px;
    padding: 9px 13px;
    margin-bottom: 5px;
    color: #1E3A8A;
    font-size: clamp(11px,1.2vw,13px);
    width: fit-content;
    max-width: 94%;
    margin-left: auto;
    word-break: break-word;
}
.chat-bubble-agent {
    background: linear-gradient(135deg,#F1F5F9,#F8FAFC);
    border: 1px solid #E2E8F0;
    border-radius: 16px 16px 16px 4px;
    padding: 9px 13px;
    margin-bottom: 10px;
    color: #334155;
    font-size: clamp(11px,1.2vw,13px);
    max-width: 94%;
    word-break: break-word;
}
.chat-label           { font-size: 10px; font-weight: 700; margin-bottom: 2px; }
.chat-user-label      { color: #3B82F6; text-align: right; }
.chat-agent-label     { color: #64748B; }
 
/* ── Param tags ────────────────────────────────────────────────── */
.param-good     { color:#16A34A; font-weight:700; }
.param-moderate { color:#D97706; font-weight:700; }
.param-concern  { color:#DC2626; font-weight:700; }
 
/* ── Tip banner ────────────────────────────────────────────────── */
.tip-banner {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 10px;
    padding: 7px 12px;
    font-size: clamp(10px,1.1vw,12px);
    color: #166534;
    margin-bottom: 10px;
    line-height: 1.5;
}
 
/* ── Responsive: tablets (≤ 1024px) ───────────────────────────── */
@media (max-width: 1024px) {
    .app-header-badges { display: none; }
    .block-container   { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
}
 
/* ── Responsive: mobile (≤ 768px) ─────────────────────────────── */
@media (max-width: 768px) {
    .app-header        { padding: 8px 12px; }
    .app-title         { font-size: 15px; }
    .app-subtitle      { display: none; }
    .panel             { padding: 10px; border-radius: 10px; }
    .panel-title       { font-size: 13px; }
    .risk-big          { font-size: 20px; }
    .chat-scroll       { max-height: 220px; }
    .block-container   { padding-left: 0.25rem !important; padding-right: 0.25rem !important; }
}
 
/* ── Streamlit widget font scaling ────────────────────────────── */
.stSlider label, .stNumberInput label, .stTextArea label,
.stButton button, .stMetric label, .stMetric [data-testid="stMetricValue"] {
    font-size: clamp(11px, 1.2vw, 14px) !important;
}
.stButton > button {
    width: 100%;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 8px 12px !important;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
 
/* Primary search button */
div[data-testid="column"] .stButton > button:first-child,
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    color: white !important;
    border: none !important;
}
 
/* Make expanders compact */
.streamlit-expanderHeader { font-size: clamp(11px,1.2vw,13px) !important; }
 
/* Dataframe responsive */
[data-testid="stDataFrame"] { width: 100% !important; }
 
/* Map container full width */
.folium-map { width: 100% !important; }
 
/* Info/warning boxes */
.stAlert { border-radius: 10px !important; font-size: clamp(11px,1.2vw,13px) !important; }
</style>
""", unsafe_allow_html=True)
 
 
# ============================================================
# HEADER  –  sticky, responsive
# ============================================================
 
st.markdown("""
<div class="app-header">
  <div class="app-header-left">
    <span class="app-header-icon">💧</span>
    <div>
      <div class="app-title">Water Quality AI Map</div>
      <div class="app-subtitle">Inputs &nbsp;·&nbsp; Interactive Map &nbsp;·&nbsp; Smart Chat Agent</div>
    </div>
  </div>
  <div class="app-header-badges">
    <span class="badge badge-blue">🛰️ Planetary Computer</span>
    <span class="badge badge-green">🤖 ML Risk Model</span>
    <span class="badge badge-orange">💬 City Chat Agent</span>
  </div>
</div>
""", unsafe_allow_html=True)
 
 
# ============================================================
# LOAD CSV
# ============================================================
 
if not os.path.exists(DATA_FILE):
    st.error("Missing file: data/final_df_water_quality.csv")
    st.stop()
 
final_df = pd.read_csv(DATA_FILE)
 
if "Unnamed: 0" in final_df.columns:
    final_df = final_df.drop(columns=["Unnamed: 0"])
 
required_cols = ["city", "pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD", "Risk_Level", "latitude", "longitude"]
for col in required_cols:
    if col not in final_df.columns:
        st.error(f"Missing required column in CSV: {col}")
        st.stop()
 
numeric_cols = [
    "temperature_2m", "temperature_2m_C", "dewpoint_temperature_2m", "dewpoint_temperature_2m_C",
    "total_precipitation_sum", "surface_pressure",
    "pH", "TDS", "TH", "Ca", "Mg", "DO", "BOD",
    "Water_Quality_Risk_Score", "Risk_Score", "Color_Intensity", "latitude", "longitude",
]
for col in numeric_cols:
    if col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
 
final_df = final_df.dropna(subset=["latitude", "longitude"])
 
# Build lowercase city lookup for fast fuzzy matching
city_lookup = {row["city"].lower().strip(): idx for idx, row in final_df.iterrows()}
all_city_names = [row["city"] for _, row in final_df.iterrows()]
 
 
# ============================================================
# CITY FUZZY MATCH
# ============================================================
 
def find_city_in_text(text: str):
    """
    Return (city_name, df_row) if any city from the dataset is mentioned
    in the user's message (case-insensitive, partial match allowed).
    Returns (None, None) if no city found.
    """
    text_lower = text.lower()
    best_city = None
    best_row  = None
    best_len  = 0
 
    for city_lower, idx in city_lookup.items():
        # Match whole word / city name inside the text
        if re.search(r'\b' + re.escape(city_lower) + r'\b', text_lower):
            if len(city_lower) > best_len:
                best_len  = len(city_lower)
                best_city = final_df.loc[idx, "city"]
                best_row  = final_df.loc[idx]
 
    return best_city, best_row
 
 
# ============================================================
# FULL DATAFRAME → LLM CONTEXT
# ============================================================
 
def dataframe_to_llm_context(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        rows.append(f"""
City: {row.get('city','Unknown')}
- pH:{row.get('pH','NA')}  TDS:{row.get('TDS','NA')} mg/L  TH:{row.get('TH','NA')} mg/L
- Ca:{row.get('Ca','NA')} mg/L  Mg:{row.get('Mg','NA')} mg/L
- DO:{row.get('DO','NA')} mg/L  BOD:{row.get('BOD','NA')} mg/L
- Risk Level:{row.get('Risk_Level','NA')}  Risk Score:{row.get('Risk_Score',row.get('Water_Quality_Risk_Score','NA'))}
- Concern Parameters:{row.get('Concern_Parameters','NA')}
- Temp:{row.get('temperature_2m_C','NA')}°C  Precip:{row.get('total_precipitation_sum','NA')}
""")
    return "\n".join(rows)
 
llm_context = dataframe_to_llm_context(final_df)
 
 
# ============================================================
# PLANETARY COMPUTER
# ============================================================
 
@st.cache_resource
def get_pc_catalog():
    if not STAC_READY:
        return None, "pystac_client / planetary_computer not installed."
    try:
        catalog = pystac_client.Client.open(
            PC_STAC_API, modifier=planetary_computer.sign_inplace,
        )
        return catalog, "Planetary Computer STAC catalog connected."
    except Exception as exc:
        return None, str(exc)
 
pc_catalog, pc_message = get_pc_catalog()
 
 
def _bbox_from_point(lat, lon, radius_m):
    deg_lat = radius_m / 111_320
    deg_lon = radius_m / (111_320 * math.cos(math.radians(lat)))
    return [lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat]
 
 
def extract_pc_data(latitude, longitude, radius_m):
    empty = {
        "altitude_m": "NA", "temperature_2m_C": "NA",
        "dewpoint_temperature_2m_C": "NA", "total_precipitation_sum": "NA",
        "surface_pressure": "NA", "ndvi_mean": "NA",
        "stac_items_found": 0, "source": "Planetary Computer",
    }
    if pc_catalog is None or not STAC_READY:
        empty["source"] = "Planetary Computer not available"
        return empty
 
    bbox   = _bbox_from_point(latitude, longitude, max(radius_m, 5_000))
    result = dict(empty)
 
    # DEM
    try:
        items = list(pc_catalog.search(collections=["cop-dem-glo-30"], bbox=bbox, max_items=4).items())
        result["stac_items_found"] += len(items)
        if items:
            import stackstac as sc
            stack = sc.stack(items, epsg=4326, resolution=0.0003, bounds=bbox, dtype="float32", fill_value=np.nan)
            valid = stack.values[~np.isnan(stack.values)]
            if valid.size > 0:
                result["altitude_m"] = round(float(np.nanmean(valid)), 2)
    except Exception as exc:
        result["altitude_m"] = f"DEM error: {exc}"
 
    # ERA5
    try:
        era5_items = list(pc_catalog.search(
            collections=["era5-pds"], bbox=bbox,
            datetime="2023-01-01/2023-01-31", max_items=10,
        ).items())
        result["stac_items_found"] += len(era5_items)
        temps, dewps, precips, pressures = [], [], [], []
 
        for item in era5_items:
            def _read(key):
                if key not in item.assets: return None
                try:
                    with rasterio.open(item.assets[key].href) as src:
                        win  = rasterio.windows.from_bounds(*bbox, transform=src.transform)
                        data = src.read(1, window=win, resampling=Resampling.bilinear)
                        vd   = data[data != src.nodata] if src.nodata else data.flatten()
                        return float(np.nanmean(vd)) if vd.size else None
                except Exception:
                    return None
 
            t = _read("2m_temperature");      t and temps.append(t)
            d = _read("2m_dewpoint_temperature"); d and dewps.append(d)
            p = _read("total_precipitation"); p and precips.append(p)
            s = _read("surface_pressure");    s and pressures.append(s)
 
        def _k2c(vals):
            if not vals: return "NA"
            m = float(np.mean(vals))
            return round(m - 273.15, 2) if m > 100 else round(m, 2)
 
        result["temperature_2m_C"]          = _k2c(temps)
        result["dewpoint_temperature_2m_C"] = _k2c(dewps)
        result["total_precipitation_sum"]   = round(float(np.mean(precips)), 6) if precips else "NA"
        result["surface_pressure"]          = round(float(np.mean(pressures)), 2) if pressures else "NA"
        result["source"] = "Copernicus DEM 30 + ERA5-PDS via Planetary Computer STAC"
    except Exception as exc:
        result["source"] = f"ERA5 query error: {exc}"
 
    # Sentinel-2 NDVI
    try:
        s2_items = list(pc_catalog.search(
            collections=["sentinel-2-l2a"], bbox=bbox,
            datetime="2023-01-01/2023-03-31",
            query={"eo:cloud_cover": {"lt": 20}}, max_items=3,
        ).items())
        result["stac_items_found"] += len(s2_items)
        if s2_items:
            import stackstac as sc
            s2 = sc.stack(s2_items, assets=["B04","B08"], epsg=4326, resolution=0.0001,
                          bounds=bbox, dtype="float32", fill_value=np.nan)
            red  = s2.sel(band="B04").values.astype(float)
            nir  = s2.sel(band="B08").values.astype(float)
            denom = nir + red
            denom[denom == 0] = np.nan
            ndvi = (nir - red) / denom
            valid = ndvi[~np.isnan(ndvi)]
            if valid.size > 0:
                result["ndvi_mean"] = round(float(np.nanmean(valid)), 4)
    except Exception:
        pass
 
    return result
 
 
# ============================================================
# UTILITIES
# ============================================================
 
def circle_area_km2(r): return math.pi * (r / 1000) ** 2
 
def get_circle_from_geometry(drawing):
    if not drawing: return None, None, None
    geom  = drawing.get("geometry", {})
    props = drawing.get("properties", {})
    if geom.get("type") == "Point" and len(geom.get("coordinates", [])) == 2:
        lon, lat = geom["coordinates"]
        return float(lat), float(lon), float(props.get("radius", 1000.0))
    return None, None, None
 
def calculate_rule_risk(row):
    score, reasons = 0, []
    pH = float(row["pH"]); TDS = float(row["TDS"]); TH = float(row["TH"])
    DO = float(row["DO"]); BOD = float(row["BOD"])
    if pH < 6.5 or pH > 8.5:  score += 2; reasons.append("pH outside range")
    if TDS > 500:              score += 1; reasons.append("high TDS")
    if TDS > 1000:             score += 2; reasons.append("very high TDS")
    if TH > 300:               score += 1; reasons.append("high hardness")
    if DO < 5:                 score += 2; reasons.append("low dissolved oxygen")
    if BOD > 3:                score += 1; reasons.append("high BOD")
    if BOD > 6:                score += 2; reasons.append("very high BOD")
    level = (
        "Very Low" if score <= 1 else "Low" if score <= 3 else
        "Moderate" if score <= 5 else "High" if score <= 7 else "Very High"
    )
    if not reasons: reasons.append("all parameters within low-risk range")
    return level, score, "; ".join(reasons)
 
def parameter_status(param, value):
    v = float(value)
    if param == "pH":
        return ("Good","Within 6.5–8.5") if 6.5 <= v <= 8.5 else ("Concern","Outside 6.5–8.5")
    if param == "TDS":
        return ("Good","Low solids") if v<=500 else ("Moderate","Elevated") if v<=1000 else ("High","Very high")
    if param == "TH":
        return ("Good","Acceptable") if v <= 300 else ("Concern","Hard water")
    if param == "DO":
        return ("Good","Healthy O₂") if v>=6 else ("Moderate","Borderline") if v>=5 else ("Concern","Low O₂")
    if param == "BOD":
        return ("Good","Low load") if v<=3 else ("Moderate","Some pollution") if v<=6 else ("High","High pollution")
    return ("Info","Mineral")
 
def haversine_km(lat1,lon1,lat2,lon2):
    R=6371.0; lat1,lon1,lat2,lon2 = map(math.radians,[float(lat1),float(lon1),float(lat2),float(lon2)])
    dlat,dlon=lat2-lat1,lon2-lon1
    a=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))
 
def find_nearest_dataframe_record(latitude, longitude):
    df = final_df.copy()
    df["distance_km"] = df.apply(lambda r: haversine_km(latitude,longitude,r["latitude"],r["longitude"]),axis=1)
    return df.sort_values("distance_km").iloc[0]
 
def add_map_legend(m):
    html = """<div style="position:fixed;bottom:35px;left:35px;width:185px;z-index:9999;
    background:white;border:2px solid #999;border-radius:10px;padding:12px;font-size:14px;
    box-shadow:2px 2px 8px rgba(0,0,0,0.25);"><b>Risk Legend</b><br>
    <span style="color:#2E7D32;">●</span> Very Low<br>
    <span style="color:#66BB6A;">●</span> Low<br>
    <span style="color:#FB8C00;">●</span> Moderate<br>
    <span style="color:#E53935;">●</span> High<br>
    <span style="color:#7F0000;">●</span> Very High</div>"""
    m.get_root().html.add_child(folium.Element(html))
 
 
# ============================================================
# ML MODEL
# ============================================================
 
ML_FEATURES = [
    "latitude","longitude","pH","TDS","TH","Ca","Mg","DO","BOD",
    "temperature_2m_C","dewpoint_temperature_2m_C","total_precipitation_sum","surface_pressure",
]
 
@st.cache_resource
def train_ml_model(df):
    t = df.copy()
    for col in ML_FEATURES:
        if col not in t.columns: t[col] = 0
        t[col] = pd.to_numeric(t[col], errors="coerce")
    t[ML_FEATURES] = t[ML_FEATURES].fillna(0)
    le = LabelEncoder()
    y  = le.fit_transform(t["Risk_Level"].astype(str))
    model = RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced")
    model.fit(t[ML_FEATURES], y)
    return model, le
 
ml_model, label_encoder = train_ml_model(final_df)
 
def predict_ml_risk(input_data, earth_data):
    row = {
        "latitude":input_data["latitude"],"longitude":input_data["longitude"],
        "pH":input_data["pH"],"TDS":input_data["TDS"],"TH":input_data["TH"],
        "Ca":input_data["Ca"],"Mg":input_data["Mg"],"DO":input_data["DO"],"BOD":input_data["BOD"],
        "temperature_2m_C":earth_data.get("temperature_2m_C"),
        "dewpoint_temperature_2m_C":earth_data.get("dewpoint_temperature_2m_C"),
        "total_precipitation_sum":earth_data.get("total_precipitation_sum"),
        "surface_pressure":earth_data.get("surface_pressure"),
    }
    pred_df = pd.DataFrame([row])
    for col in ML_FEATURES:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
    pred_df[ML_FEATURES] = pred_df[ML_FEATURES].fillna(0)
    code  = ml_model.predict(pred_df[ML_FEATURES])[0]
    level = label_encoder.inverse_transform([code])[0]
    probs = ml_model.predict_proba(pred_df[ML_FEATURES])[0]
    prob_df = pd.DataFrame({"Risk_Level":label_encoder.classes_,"Probability":probs}).sort_values("Probability",ascending=False)
    return level, prob_df
 
 
# ============================================================
# EXPLANATION GENERATOR
# ============================================================
 
def generate_explanation(input_data, earth_data, nearest_record,
                         ml_risk_level, rule_risk_level, rule_score,
                         color, intensity, concerns, probability_df):
    ndvi = earth_data.get("ndvi_mean","NA")
    ndvi_note = ""
    try:
        nv = float(ndvi)
        if nv < 0.2:   ndvi_note = "Sparse/no vegetation — possible urban or bare soil."
        elif nv < 0.4: ndvi_note = "Mixed land use — moderate agricultural runoff risk."
        else:          ndvi_note = "Dense vegetation — potential fertiliser/pesticide runoff."
    except Exception:
        pass
 
    return f"""
<div class="risk-card" style="background:{color};">
    <div class="risk-big">{RISK_STYLE[ml_risk_level]['emoji']} {ml_risk_level} Risk</div>
    <div class="risk-small">ML · Planetary Computer STAC + water-quality dataframe</div>
    <div class="risk-small">Rule score: {rule_score} | Intensity: {intensity}</div>
    <div class="risk-small">⚠️ Concerns: {concerns}</div>
    {"<div class='risk-small'>🌿 NDVI: " + str(ndvi) + " — " + ndvi_note + "</div>" if ndvi_note else ""}
</div>
 
**Nearest reference city:** {nearest_record.get('city','Unknown')} &nbsp;
({nearest_record.get('distance_km',0):.2f} km away)
 
**Why this risk level?**
The Random Forest model learned from `final_df_water_quality.csv`. The selected area's
parameters were compared against city-level patterns. Key concern parameters: **{concerns}**.
 
| Parameter | Value | Status |
|-----------|-------|--------|
| pH | {input_data['pH']} | {"✅" if 6.5<=float(input_data['pH'])<=8.5 else "⚠️"} |
| TDS | {input_data['TDS']} mg/L | {"✅" if float(input_data['TDS'])<=500 else "⚠️"} |
| Total Hardness | {input_data['TH']} mg/L | {"✅" if float(input_data['TH'])<=300 else "⚠️"} |
| Dissolved O₂ | {input_data['DO']} mg/L | {"✅" if float(input_data['DO'])>=6 else "⚠️"} |
| BOD | {input_data['BOD']} mg/L | {"✅" if float(input_data['BOD'])<=3 else "⚠️"} |
 
**Altitude:** {earth_data.get('altitude_m')} m &nbsp; **Temp:** {earth_data.get('temperature_2m_C')} °C &nbsp;
**Precip:** {earth_data.get('total_precipitation_sum')} &nbsp; **NDVI:** {ndvi}
"""
 
 
# ============================================================
# SMART CHAT AGENT
# ============================================================
 
def _city_card(row) -> str:
    """Format a city's water quality summary."""
    risk = row.get("Risk_Level", "NA")
    emoji = RISK_STYLE.get(str(risk), {}).get("emoji", "⬜")
    return (
        f"**{row.get('city','Unknown')}** {emoji} {risk} risk\n"
        f"- pH: {row.get('pH','NA')} | TDS: {row.get('TDS','NA')} mg/L | "
        f"TH: {row.get('TH','NA')} mg/L\n"
        f"- DO: {row.get('DO','NA')} mg/L | BOD: {row.get('BOD','NA')} mg/L\n"
        f"- Ca: {row.get('Ca','NA')} mg/L | Mg: {row.get('Mg','NA')} mg/L"
    )
 
 
def answer_chatbot(question: str) -> tuple[str, dict | None]:
    """
    Returns (answer_text, map_update_dict | None).
    map_update_dict = {"latitude": ..., "longitude": ..., "city": ...} if map should move.
    """
    q   = question.lower().strip()
    ss  = st.session_state
 
    # ── 1. City navigation intent ──────────────────────────────────────────
    nav_patterns = [
        r"(go to|navigate to|show|move to|change.*?to|point.*?to|set.*?to|focus on|zoom to|take me to|look at)\s+(.+)",
        r"(what about|tell me about|show me)\s+(.+)",
        r"can (?:you |the )?(?:map )?(?:be |show |change to |point to )?(.+)\??$",
    ]
    city_name, city_row = find_city_in_text(q)
 
    nav_intent = any(
        kw in q for kw in [
            "go to","navigate","show","move","change","point","focus","zoom",
            "take me","look at","what about","tell me about","can the map",
            "map to","map point","change map",
        ]
    )
 
    if city_name and city_row is not None:
        lat = float(city_row["latitude"])
        lon = float(city_row["longitude"])
        map_update = {"latitude": lat, "longitude": lon, "city": city_name}
 
        card = _city_card(city_row)
        reply = (
            f"📍 Moving map to **{city_name}** (lat: {lat:.4f}, lon: {lon:.4f}).\n\n"
            f"{card}\n\n"
            f"The map centre has been updated. Click **Search Water Quality** to run the ML prediction for this location."
        )
        return reply, map_update
 
    # ── 2. List best / safest cities ──────────────────────────────────────
    if any(kw in q for kw in ["best","safest","cleanest","good place","good water","safe to drink","lowest risk"]):
        good = final_df[final_df["Risk_Level"].isin(["Very Low","Low"])].copy()
        if good.empty:
            good = final_df.sort_values("Risk_Score" if "Risk_Score" in final_df.columns else "BOD").head(5)
        good = good.drop_duplicates("city").head(5)
        lines = [f"{i+1}. {_city_card(r)}" for i, (_,r) in enumerate(good.iterrows())]
        return "Here are the cities with the best water quality in the dataset:\n\n" + "\n\n".join(lines), None
 
    # ── 3. List worst / most polluted cities ──────────────────────────────
    if any(kw in q for kw in ["worst","most polluted","highest risk","dangerous","bad water"]):
        bad = final_df[final_df["Risk_Level"].isin(["Very High","High"])].copy()
        if bad.empty:
            bad = final_df.sort_values("BOD", ascending=False).head(5)
        bad = bad.drop_duplicates("city").head(5)
        lines = [f"{i+1}. {_city_card(r)}" for i, (_,r) in enumerate(bad.iterrows())]
        return "Cities with the highest water quality risk in the dataset:\n\n" + "\n\n".join(lines), None
 
    # ── 4. List all cities ─────────────────────────────────────────────────
    if any(kw in q for kw in ["list cities","all cities","which cities","available cities","show cities"]):
        cities_by_risk = final_df.drop_duplicates("city").sort_values("Risk_Level")
        lines = []
        for _, r in cities_by_risk.iterrows():
            emoji = RISK_STYLE.get(str(r.get("Risk_Level","")), {}).get("emoji","⬜")
            lines.append(f"{emoji} **{r['city']}** — {r.get('Risk_Level','NA')} risk")
        return "**Cities in the dataset:**\n\n" + "\n".join(lines), None
 
    # ── 5. Compare two cities ──────────────────────────────────────────────
    if any(kw in q for kw in ["compare","vs","versus","difference between","better than"]):
        found_cities = []
        for city_lower, idx in city_lookup.items():
            if re.search(r'\b' + re.escape(city_lower) + r'\b', q):
                found_cities.append(final_df.loc[idx])
        if len(found_cities) >= 2:
            r1, r2 = found_cities[0], found_cities[1]
            return (
                f"**Comparison: {r1['city']} vs {r2['city']}**\n\n"
                f"| Parameter | {r1['city']} | {r2['city']} |\n"
                f"|-----------|-----------|----------|\n"
                f"| pH | {r1.get('pH','NA')} | {r2.get('pH','NA')} |\n"
                f"| TDS (mg/L) | {r1.get('TDS','NA')} | {r2.get('TDS','NA')} |\n"
                f"| TH (mg/L) | {r1.get('TH','NA')} | {r2.get('TH','NA')} |\n"
                f"| DO (mg/L) | {r1.get('DO','NA')} | {r2.get('DO','NA')} |\n"
                f"| BOD (mg/L) | {r1.get('BOD','NA')} | {r2.get('BOD','NA')} |\n"
                f"| Risk Level | {r1.get('Risk_Level','NA')} | {r2.get('Risk_Level','NA')} |"
            ), None
 
    # ── 6. Prediction-dependent questions ─────────────────────────────────
    if not ss.get("prediction_done"):
        # Still try to answer general questions without needing prediction
        if any(kw in q for kw in ["ph","tds","bod","do","oxygen","hardness","risk","quality","water"]):
            return (
                "I can answer detailed questions about the selected area after you click "
                "**Search Water Quality**. Or ask me about a specific city — e.g. "
                "*'Show me Mumbai'* or *'What is the water quality in Delhi?'*"
            ), None
        return (
            "Please draw a circle on the map, enter water-quality values, and click "
            "**Search Water Quality** — then I can answer questions about the selected area. "
            "You can also ask me to show a specific city, e.g. *'Go to Chennai'*."
        ), None
 
    input_data  = ss.input_data
    earth_data  = ss.earth_data
    nearest     = ss.nearest_record
 
    # ── 7. Water quality / risk questions ─────────────────────────────────
    if any(kw in q for kw in ["water quality","what is the quality","is the water","quality here","quality of water"]):
        risk  = ss.ml_risk_level
        color = RISK_STYLE.get(risk,{}).get("emoji","")
        return (
            f"The water quality at the selected location is predicted as "
            f"**{color} {risk} risk**.\n\n"
            f"- Rule-based score: **{ss.rule_score}** / 9\n"
            f"- Main concerns: **{ss.concerns}**\n"
            f"- Nearest reference city: **{nearest.get('city','Unknown')}** "
            f"({nearest.get('distance_km',0):.2f} km away, "
            f"{nearest.get('Risk_Level','NA')} risk)"
        ), None
 
    if any(kw in q for kw in ["risk","prediction","result","level","danger"]):
        return (
            f"**ML prediction: {RISK_STYLE.get(ss.ml_risk_level,{}).get('emoji','')} {ss.ml_risk_level} risk**\n\n"
            f"Rule-based score: {ss.rule_score} | Concerns: {ss.concerns}"
        ), None
 
    # ── 8. Individual parameter questions ─────────────────────────────────
    if "ph" in q:
        v = input_data['pH']; s, m = parameter_status("pH", v)
        return f"**pH = {v}** — {m}. {'✅ Good' if s=='Good' else '⚠️ ' + s}. Recommended range: 6.5–8.5.", None
 
    if "tds" in q:
        v = input_data['TDS']; s, m = parameter_status("TDS", v)
        return f"**TDS = {v} mg/L** — {m}. {'✅ Good' if s=='Good' else '⚠️ ' + s}. Below 500 mg/L is ideal.", None
 
    if any(kw in q for kw in ["hardness","total hardness"," th "]):
        v = input_data['TH']; s, m = parameter_status("TH", v)
        return f"**Total Hardness = {v} mg/L** — {m}. {'✅ Good' if s=='Good' else '⚠️ ' + s}. Below 300 mg/L is acceptable.", None
 
    if any(kw in q for kw in ["bod","biological oxygen","biochemical"]):
        v = input_data['BOD']; s, m = parameter_status("BOD", v)
        return f"**BOD = {v} mg/L** — {m}. {'✅ Good' if s=='Good' else '⚠️ ' + s}. Below 3 mg/L indicates clean water.", None
 
    if any(kw in q for kw in [" do ", "dissolved oxygen","oxygen level"]):
        v = input_data['DO']; s, m = parameter_status("DO", v)
        return f"**Dissolved Oxygen = {v} mg/L** — {m}. {'✅ Good' if s=='Good' else '⚠️ ' + s}. Above 6 mg/L is healthy.", None
 
    if any(kw in q for kw in ["calcium"," ca "]):
        return f"**Calcium (Ca) = {input_data['Ca']} mg/L** — mineral component contributing to water hardness.", None
 
    if any(kw in q for kw in ["magnesium"," mg "]):
        return f"**Magnesium (Mg) = {input_data['Mg']} mg/L** — mineral component contributing to water hardness.", None
 
    # ── 9. Environmental / satellite data ─────────────────────────────────
    if any(kw in q for kw in ["altitude","elevation","dem","height"]):
        return f"**Elevation = {earth_data.get('altitude_m')} m** (Copernicus DEM 30 via Planetary Computer).", None
 
    if any(kw in q for kw in ["temperature","temp","climate","weather"]):
        return (
            f"**Temperature = {earth_data.get('temperature_2m_C')} °C** (ERA5-PDS, Jan 2023)\n"
            f"Dewpoint = {earth_data.get('dewpoint_temperature_2m_C')} °C | "
            f"Precipitation = {earth_data.get('total_precipitation_sum')} | "
            f"Pressure = {earth_data.get('surface_pressure')} Pa"
        ), None
 
    if any(kw in q for kw in ["ndvi","vegetation","greenness","plant"]):
        ndvi = earth_data.get("ndvi_mean","NA")
        try:
            nv = float(ndvi)
            note = "sparse vegetation" if nv<0.2 else "moderate vegetation" if nv<0.4 else "dense vegetation"
            return f"**NDVI = {ndvi}** — {note} (Sentinel-2 L2A). Higher NDVI → more agricultural runoff risk.", None
        except Exception:
            return f"NDVI data not available for this location (value: {ndvi}).", None
 
    # ── 10. Nearest city ────────────────────────────────────────────────────
    if any(kw in q for kw in ["nearest","closest","near","reference city","nearby"]):
        return (
            f"The nearest reference city is **{nearest.get('city','Unknown')}**, "
            f"**{nearest.get('distance_km',0):.2f} km** from the selected circle centre.\n\n"
            f"{_city_card(nearest)}"
        ), None
 
    # ── 11. Circle / area info ──────────────────────────────────────────────
    if any(kw in q for kw in ["area","radius","circle","location","coordinates","lat","lon"]):
        return (
            f"**Selected circle:**\n"
            f"- Centre: {input_data['latitude']:.6f}, {input_data['longitude']:.6f}\n"
            f"- Radius: {input_data['radius_m']} m\n"
            f"- Area: {input_data['area_km2']:.3f} km²"
        ), None
 
    # ── 12. Planetary Computer / STAC ──────────────────────────────────────
    if any(kw in q for kw in ["planetary","stac","copernicus","era5","sentinel","satellite"]):
        return (
            "The app queries **Microsoft Planetary Computer** via `pystac_client` and `planetary_computer`.\n\n"
            "| Collection | Data |\n|---|---|\n"
            "| `cop-dem-glo-30` | Terrain elevation |\n"
            "| `era5-pds` | Temperature, precipitation, pressure |\n"
            "| `sentinel-2-l2a` | NDVI / vegetation |\n\n"
            "No API key needed — assets are signed automatically."
        ), None
 
    # ── 13. Summary / help ──────────────────────────────────────────────────
    if any(kw in q for kw in ["help","what can","summary","overview","tell me everything"]):
        return (
            "**I can answer:**\n"
            "- 📍 *'Go to Mumbai'* / *'Show Delhi on the map'* → moves the map\n"
            "- 💧 *'What is the water quality?'* → risk level & concerns\n"
            "- 📊 *'What is the pH / TDS / BOD / DO?'* → parameter details\n"
            "- 🏙️ *'Which city is nearest?'* → reference city\n"
            "- 🌿 *'Best cities for water quality?'* → top safe cities\n"
            "- ⚠️ *'Worst / most polluted cities?'* → high-risk cities\n"
            "- 🔬 *'Compare Mumbai and Delhi'* → side-by-side table\n"
            "- 🛰️ *'What is the NDVI / altitude / temperature?'* → satellite data"
        ), None
 
    # ── 14. Fallback ────────────────────────────────────────────────────────
    return (
        f"The selected area shows **{ss.ml_risk_level} risk** "
        f"(nearest city: **{nearest.get('city','Unknown')}**, {nearest.get('distance_km',0):.2f} km away).\n\n"
        f"Concerns: {ss.concerns}.\n\n"
        "Try asking: *'What is the water quality?'*, *'Go to Mumbai'*, *'Best cities?'*, or *'Compare Delhi and Chennai'*."
    ), None
 
 
# ============================================================
# SESSION STATE
# ============================================================
 
defaults = {
    "latitude":             float(final_df["latitude"].iloc[0]),
    "longitude":            float(final_df["longitude"].iloc[0]),
    "radius_m":             1000.0,
    "drawn_area":           None,
    "prediction_done":      False,
    "chat_history":         [],
    "last_input_signature": None,
    "map_moved_by_chat":    False,
}
 
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
 
 
# ============================================================
# THREE-PANEL GUI
# ============================================================
 
left_col, map_col, chat_col = st.columns([1, 2.2, 1.1], gap="small")
 
 
# ============================================================
# LEFT PANEL
# ============================================================
 
with left_col:
    st.markdown('<div class="panel"><div class="panel-title">⚗️ Water Inputs</div>', unsafe_allow_html=True)
 
    with st.expander("Circle area", expanded=True):
        latitude  = st.number_input("Center latitude",  value=float(st.session_state.latitude),  format="%.6f")
        longitude = st.number_input("Center longitude", value=float(st.session_state.longitude), format="%.6f")
        radius_m  = st.number_input("Radius meters",    value=float(st.session_state.radius_m),  min_value=50.0, step=100.0)
        st.session_state.latitude  = latitude
        st.session_state.longitude = longitude
        st.session_state.radius_m  = radius_m
        st.caption(f"Area: {circle_area_km2(radius_m):.3f} km²")
 
    pH  = st.slider("pH",                       0.0, 14.0,  7.2, 0.1)
    TDS = st.number_input("TDS mg/L",           value=450.0, step=10.0)
    TH  = st.number_input("Total Hardness mg/L",value=180.0, step=10.0)
    Ca  = st.number_input("Calcium mg/L",       value=60.0,  step=1.0)
    Mg  = st.number_input("Magnesium mg/L",     value=25.0,  step=1.0)
    DO  = st.slider("Dissolved Oxygen mg/L",    0.0, 15.0,  6.5, 0.1)
    BOD = st.slider("BOD mg/L",                 0.0, 20.0,  3.2, 0.1)
 
    input_data = {
        "latitude":latitude,"longitude":longitude,"radius_m":radius_m,
        "area_km2":circle_area_km2(radius_m),
        "pH":pH,"TDS":TDS,"TH":TH,"Ca":Ca,"Mg":Mg,"DO":DO,"BOD":BOD,
    }
 
    current_sig = tuple(input_data.values())
    if st.session_state.last_input_signature is None:
        st.session_state.last_input_signature = current_sig
    if current_sig != st.session_state.last_input_signature:
        st.session_state.prediction_done      = False
        st.session_state.last_input_signature = current_sig
 
    search_clicked = st.button("🔍 Search Water Quality", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
 
    # Legend
    st.markdown('<div class="panel"><div class="panel-title">Legend</div>', unsafe_allow_html=True)
    for level, info in RISK_STYLE.items():
        st.markdown(
            f'<div class="legend-row">'
            f'<span class="legend-dot" style="background:{info["color"]};"></span>'
            f'{info["emoji"]} {level}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
 
 
# ============================================================
# MAP PANEL
# ============================================================
 
with map_col:
    st.markdown('<div class="panel"><div class="panel-title">🗺️ Interactive Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint-box">✏️ <b>Draw a circle</b> on the map — or type '
        '<b>"Go to Mumbai"</b> in the chat to navigate instantly.</div>',
        unsafe_allow_html=True,
    )
 
    m = folium.Map(
        location=[st.session_state.latitude, st.session_state.longitude],
        zoom_start=6, tiles="OpenStreetMap",
    )
 
    Draw(
        export=True,
        draw_options={"polyline":False,"rectangle":False,"polygon":False,
                      "circle":True,"marker":False,"circlemarker":False},
        edit_options={"edit":True,"remove":True},
    ).add_to(m)
 
    # Show all dataset cities as small markers
    for _, row in final_df.drop_duplicates("city").iterrows():
        risk    = str(row.get("Risk_Level","NA"))
        clr     = RISK_STYLE.get(risk, {}).get("color","#999")
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=5, color=clr, fill=True, fill_color=clr, fill_opacity=0.7,
            tooltip=f"{row['city']} — {risk}",
            popup=folium.Popup(
                f"<b>{row['city']}</b><br>Risk: {risk}<br>"
                f"pH:{row.get('pH','NA')} TDS:{row.get('TDS','NA')}<br>"
                f"DO:{row.get('DO','NA')} BOD:{row.get('BOD','NA')}",
                max_width=200,
            ),
        ).add_to(m)
 
    # Selected circle
    folium.Circle(
        location=[st.session_state.latitude, st.session_state.longitude],
        radius=st.session_state.radius_m,
        color="#2563EB", fill=True, fill_color="#93C5FD", fill_opacity=0.22,
        popup="Selected circle area",
    ).add_to(m)
 
    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        popup="Circle centre", tooltip="Circle centre",
        icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
    ).add_to(m)
 
    if st.session_state.prediction_done:
        folium.Circle(
            location=[st.session_state.latitude, st.session_state.longitude],
            radius=st.session_state.radius_m,
            color=st.session_state.map_color,
            fill=True, fill_color=st.session_state.map_color,
            fill_opacity=st.session_state.intensity,
            popup=f"{st.session_state.ml_risk_level} Risk",
        ).add_to(m)
 
    add_map_legend(m)
 
    map_data = st_folium(
        m, height=520, use_container_width=True,
        returned_objects=["last_active_drawing","all_drawings"],
        key="water_quality_circle_map",
    )
 
    if map_data and map_data.get("last_active_drawing"):
        st.session_state.drawn_area = map_data["last_active_drawing"]
        lat_, lon_, drawn_r = get_circle_from_geometry(st.session_state.drawn_area)
        if lat_ is not None:
            st.session_state.latitude  = lat_
            st.session_state.longitude = lon_
            st.session_state.radius_m  = drawn_r
            st.session_state.prediction_done = False
 
    st.info(
        f"📍 Centre: {st.session_state.latitude:.6f}, {st.session_state.longitude:.6f} | "
        f"Radius: {st.session_state.radius_m:.1f} m | "
        f"Area: {circle_area_km2(st.session_state.radius_m):.3f} km²"
    )
    st.markdown("</div>", unsafe_allow_html=True)
 
 
# ============================================================
# CHAT PANEL
# ============================================================
 
with chat_col:
    st.markdown('<div class="panel"><div class="panel-title">🤖 Chat Agent</div>', unsafe_allow_html=True)
 
    st.markdown(
        '<div class="tip-banner">'
        '💡 <b>"Go to Mumbai"</b> &nbsp;·&nbsp; <b>"Best cities?"</b> &nbsp;·&nbsp; '
        '<b>"Water quality here?"</b> &nbsp;·&nbsp; <b>"Compare Delhi &amp; Chennai"</b>'
        '</div>',
        unsafe_allow_html=True,
    )
 
    question    = st.text_area(
        "Ask about water quality or a city:",
        placeholder="e.g. 'Go to Mumbai' or 'What is the BOD?'",
        height=80,
        key="chat_input",
    )
    ask_clicked = st.button("💬 Ask Agent", use_container_width=True)
 
    if ask_clicked and question.strip():
        answer, map_update = answer_chatbot(question)
 
        if map_update:
            st.session_state.latitude  = map_update["latitude"]
            st.session_state.longitude = map_update["longitude"]
            st.session_state.prediction_done   = False
            st.session_state.map_moved_by_chat = True
 
        st.session_state.chat_history.append({
            "question": question,
            "answer":   answer,
            "map_update": map_update,
        })
        st.rerun()
 
    # Scrollable chat history
    if st.session_state.chat_history:
        chat_html = '<div class="chat-scroll">'
        for chat in reversed(st.session_state.chat_history[-8:]):
            mu = chat.get("map_update")
            chat_html += (
                f'<div class="chat-label chat-user-label">You</div>'
                f'<div class="chat-bubble-user">{chat["question"]}</div>'
            )
            agent_text = chat["answer"].replace("\n", "<br>").replace("**", "<b>", 1)
            # Close any opened bold tags naively
            agent_text = agent_text.replace("**", "</b>")
            if mu:
                agent_text = f"🗺️ Map → <b>{mu['city']}</b><br><br>" + agent_text
            chat_html += (
                f'<div class="chat-label chat-agent-label">Agent</div>'
                f'<div class="chat-bubble-agent">{agent_text}</div>'
            )
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
 
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
 
    st.markdown("</div>", unsafe_allow_html=True)
 
    # System panel
    st.markdown('<div class="panel"><div class="panel-title">⚙️ System</div>', unsafe_allow_html=True)
    if STAC_READY and pc_catalog is not None:
        st.success("✅ Planetary Computer")
    elif STAC_READY:
        st.warning("⚠️ Catalog failed")
    else:
        st.warning("⚠️ STAC not installed")
        st.caption("pip install pystac-client planetary-computer stackstac rasterio")
 
    st.caption(
        f"🏙️ {len(final_df)} cities &nbsp;|&nbsp; 🌲 Random Forest &nbsp;|&nbsp; 🛰️ PC STAC"
    )
    st.markdown("</div>", unsafe_allow_html=True)
 
 
# ============================================================
# SEARCH ACTION
# ============================================================
 
if search_clicked:
    with st.spinner("🛰️ Querying Planetary Computer STAC…"):
        earth_data = extract_pc_data(latitude, longitude, radius_m)
 
    nearest_record = find_nearest_dataframe_record(latitude, longitude)
    rule_level, rule_score, concerns = calculate_rule_risk(input_data)
    ml_risk_level, probability_df   = predict_ml_risk(input_data, earth_data)
 
    if ml_risk_level not in RISK_STYLE:
        ml_risk_level = rule_level
 
    map_color = RISK_STYLE[ml_risk_level]["color"]
    intensity = RISK_STYLE[ml_risk_level]["intensity"]
 
    explanation = generate_explanation(
        input_data, earth_data, nearest_record,
        ml_risk_level, rule_level, rule_score,
        map_color, intensity, concerns, probability_df,
    )
 
    st.session_state.update({
        "prediction_done": True,
        "input_data":      input_data,
        "earth_data":      earth_data,
        "nearest_record":  nearest_record,
        "ml_risk_level":   ml_risk_level,
        "rule_risk_level": rule_level,
        "rule_score":      rule_score,
        "map_color":       map_color,
        "intensity":       intensity,
        "concerns":        concerns,
        "explanation":     explanation,
        "probability_df":  probability_df,
    })
 
 
# ============================================================
# DASHBOARD + RESULT
# ============================================================
 
st.divider()
dash_col, result_col = st.columns([1, 2], gap="small")
 
with dash_col:
    st.markdown('<div class="panel"><div class="panel-title">📊 Input Dashboard</div>', unsafe_allow_html=True)
 
    param_rows = [
        {
            "Parameter": p,
            "Value":     input_data[p],
            "Status":    parameter_status(p, input_data[p])[0],
            "Meaning":   parameter_status(p, input_data[p])[1],
        }
        for p in ["pH","TDS","TH","Ca","Mg","DO","BOD"]
    ]
    st.dataframe(pd.DataFrame(param_rows), use_container_width=True, height=280)
    st.bar_chart(
        pd.DataFrame({
            "Parameter": ["pH","TDS","TH","Ca","Mg","DO","BOD"],
            "Value":     [pH,TDS,TH,Ca,Mg,DO,BOD],
        }).set_index("Parameter")
    )
    st.markdown("</div>", unsafe_allow_html=True)
 
with result_col:
    st.markdown('<div class="panel"><div class="panel-title">🔎 Search Result</div>', unsafe_allow_html=True)
 
    if not st.session_state.prediction_done:
        st.info("Draw a circle, enter inputs, then click **Search Water Quality**.\n\n"
                "Or ask the chat agent: *'Go to Mumbai'* to navigate to a city.")
    else:
        st.markdown(st.session_state.explanation, unsafe_allow_html=True)
 
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("ML Risk",     st.session_state.ml_risk_level)
        c2.metric("Rule Score",  st.session_state.rule_score)
        c3.metric("Color",       st.session_state.map_color)
        c4.metric("Intensity",   st.session_state.intensity)
        c5.metric("Nearest City",st.session_state.nearest_record.get("city","NA"))
 
        st.progress(min(st.session_state.rule_score / 9, 1.0))
 
        with st.expander("ML Class Probabilities"):
            st.dataframe(st.session_state.probability_df, use_container_width=True)
            st.bar_chart(st.session_state.probability_df.set_index("Risk_Level")["Probability"])
 
        with st.expander("Nearest Reference City"):
            ndf = pd.DataFrame([st.session_state.nearest_record])
            show = [c for c in ["city","distance_km","latitude","longitude",
                                 "pH","TDS","TH","Ca","Mg","DO","BOD","Risk_Level","Risk_Score"]
                    if c in ndf.columns]
            st.dataframe(ndf[show], use_container_width=True)
 
        with st.expander("🛰️ Planetary Computer STAC Data"):
            st.json(st.session_state.earth_data)
 
        with st.expander("LLM Context from Dataframe"):
            st.text(llm_context[:3000] + "\n...[truncated]" if len(llm_context) > 3000 else llm_context)
 
        result_df = pd.DataFrame([{
            "latitude":st.session_state.latitude,"longitude":st.session_state.longitude,
            "radius_m":st.session_state.radius_m,"area_km2":circle_area_km2(st.session_state.radius_m),
            "nearest_city":st.session_state.nearest_record.get("city","NA"),
            "nearest_distance_km":st.session_state.nearest_record.get("distance_km","NA"),
            "ML_Risk_Level":st.session_state.ml_risk_level,
            "Rule_Risk_Level":st.session_state.rule_risk_level,
            "Rule_Score":st.session_state.rule_score,
            "Color":st.session_state.map_color,"Intensity":st.session_state.intensity,
            "Concern_Parameters":st.session_state.concerns,
            "altitude_m":st.session_state.earth_data.get("altitude_m"),
            "temperature_2m_C":st.session_state.earth_data.get("temperature_2m_C"),
            "dewpoint_2m_C":st.session_state.earth_data.get("dewpoint_temperature_2m_C"),
            "precipitation":st.session_state.earth_data.get("total_precipitation_sum"),
            "surface_pressure":st.session_state.earth_data.get("surface_pressure"),
            "ndvi_mean":st.session_state.earth_data.get("ndvi_mean"),
            "pH":st.session_state.input_data["pH"],"TDS":st.session_state.input_data["TDS"],
            "TH":st.session_state.input_data["TH"],"Ca":st.session_state.input_data["Ca"],
            "Mg":st.session_state.input_data["Mg"],"DO":st.session_state.input_data["DO"],
            "BOD":st.session_state.input_data["BOD"],
        }])
 
        st.download_button(
            "⬇️ Download Search Result CSV",
            result_df.to_csv(index=False),
            "water_quality_search_result.csv","text/csv",
            use_container_width=True,
        )
 
    st.markdown("</div>", unsafe_allow_html=True)
 

Failed to download files

