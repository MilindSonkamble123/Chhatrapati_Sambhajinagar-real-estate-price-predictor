from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))\nfrom location_config import LOCALITY_COORDS, LOCALITIES, SAMBHAJINAGAR_LOCATIONS

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "real_data1.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "real_estate_model.pkl"
DB_PATH = PROJECT_ROOT / "data" / "real_estate_app.db"
LOGO_PATH = PROJECT_ROOT / "assets" / "csn-real-estate-logo.png"

if not DATASET_PATH.exists():
    st.error("Real dataset not found. Please copy the CSV into the data folder before running the app.")
    st.stop()

if not MODEL_PATH.exists():
    from train_model import train_and_save_model

    train_and_save_model()

model = joblib.load(MODEL_PATH)
dataset = pd.read_csv(DATASET_PATH)
dataset["Price_INR"] = pd.to_numeric(dataset["Price_INR"], errors="coerce")
dataset = dataset.dropna(subset=["Price_INR"]).copy()

FEATURE_COLUMNS = [
    "Locality",
    "Property_Type",
    "BHK",
    "Bathrooms",
    "Builtup_Area_sqft",
    "Floor",
    "Property_Age_Years",
    "Furnishing",
    "Sale_Status",
    "Amenities_Level",
    "Construction_Type",
    "Parking",
    "Gated_Community",
    "School_Distance_km",
    "Hospital_Distance_km",
    "Airport_Distance_km",
    "Railway_Distance_km",
    "Bus_Stand_Distance_km",
    "Main_Road_Distance_km",
    "Pollution_AQI_Approx",
    "Total_Rooms_Approx",
    "Area_per_BHK_sqft",
]

PROPERTY_TYPES = sorted(dataset["Property_Type"].dropna().unique().tolist())
FURNISHING_OPTIONS = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
SALE_STATUS_OPTIONS = ["Ready to Move", "Resale", "Under Construction"]
AMENITY_LEVELS = ["None", "Basic", "Standard", "Premium"]
CONSTRUCTION_TYPES = sorted(dataset["Construction_Type"].dropna().unique().tolist())

# Use centralized Sambhajinagar locality config for map selection and form synchronization.
# Keep the model working against real locality names from the dataset while exposing the map-friendly list.
LOCALITY_COORDS = dict(LOCALITY_COORDS)
LOCALITIES = sorted(set(dataset["Locality"].dropna().astype(str).unique().tolist()) | set(LOCALITY_COORDS.keys()))
LOCALITY_OPTIONS = ["Select a locality"] + LOCALITIES
DEFAULT_LOCALITY = "CIDCO N-3" if "CIDCO N-3" in LOCALITIES else (LOCALITIES[0] if LOCALITIES else "")

PROPERTY_IMAGE_LIBRARY = {
    "Apartment": "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?auto=format&fit=crop&w=1200&q=80",
    "Flat": "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=1200&q=80",
    "Villa": "https://images.unsplash.com/photo-1494526585095-c41746248156?auto=format&fit=crop&w=1200&q=80",
    "House": "https://images.unsplash.com/photo-1484154218962-a197022b5858?auto=format&fit=crop&w=1200&q=80",
    "Bungalow": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?auto=format&fit=crop&w=1200&q=80",
    "Default": "https://images.unsplash.com/photo-1568605114967-8130f3a36994?auto=format&fit=crop&w=1200&q=80",
}


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


PASSWORD_HASH_PREFIX = "pbkdf2_sha256"
PASSWORD_HASH_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("ascii"),
        PASSWORD_HASH_ITERATIONS,
    ).hex()
    return f"{PASSWORD_HASH_PREFIX}${PASSWORD_HASH_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    if stored_hash.startswith(f"{PASSWORD_HASH_PREFIX}$"):
        try:
            algorithm, iterations, salt, digest = stored_hash.split("$", 3)
            if algorithm != PASSWORD_HASH_PREFIX:
                return False
            iteration_count = int(iterations)
            if not 100_000 <= iteration_count <= 1_000_000:
                return False
            actual = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("ascii"),
                iteration_count,
            ).hex()
            return hmac.compare_digest(actual.encode("ascii"), digest.encode("ascii"))
        except (ValueError, UnicodeEncodeError):
            return False

    try:
        legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(legacy_hash.encode("ascii"), stored_hash.encode("ascii"))
    except UnicodeEncodeError:
        return False


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS property_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            locality TEXT,
            property_type TEXT,
            bhk INTEGER,
            bathrooms INTEGER,
            builtup_area_sqft REAL,
            floor INTEGER,
            property_age_years INTEGER,
            furnishing TEXT,
            sale_status TEXT,
            amenities_level TEXT,
            construction_type TEXT,
            parking INTEGER,
            gated_community INTEGER,
            school_distance_km REAL,
            hospital_distance_km REAL,
            airport_distance_km REAL,
            railway_distance_km REAL,
            bus_stand_distance_km REAL,
            main_road_distance_km REAL,
            pollution_aqi REAL,
            estimated_price_inr REAL,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    columns = [row[1] for row in conn.execute("PRAGMA table_info(property_history)").fetchall()]
    if "image_url" not in columns:
        conn.execute("ALTER TABLE property_history ADD COLUMN image_url TEXT")

    conn.commit()
    conn.close()


@st.cache_data
def get_locality_summary(locality: str):
    if locality not in dataset["Locality"].unique():
        return 0.0, 0.0, 0.0
    locality_data = dataset[dataset["Locality"] == locality]
    avg_price = float(locality_data["Price_INR"].mean())
    median_price = float(locality_data["Price_INR"].median())
    min_price = float(locality_data["Price_INR"].min())
    return avg_price, median_price, min_price


@st.cache_data
def get_market_insights():
    locality_summary = (
        dataset.groupby("Locality", as_index=False)["Price_INR"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "avg_price", "median": "median_price", "count": "listings"})
        .sort_values("avg_price", ascending=False)
        .reset_index(drop=True)
    )
    property_summary = (
        dataset.groupby("Property_Type", as_index=False)["Price_INR"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_price", "count": "listings"})
        .sort_values("avg_price", ascending=False)
        .reset_index(drop=True)
    )
    affordable_locality = locality_summary.sort_values("avg_price").iloc[0].to_dict()
    premium_locality = locality_summary.iloc[0].to_dict()
    best_property_type = property_summary.iloc[0].to_dict()
    return {
        "locality_summary": locality_summary,
        "property_summary": property_summary,
        "premium_locality": premium_locality,
        "affordable_locality": affordable_locality,
        "best_property_type": best_property_type,
    }


@st.cache_data
def get_neighborhood_map_data():
    rows = []
    for locality, (lat, lon) in LOCALITY_COORDS.items():
        avg_price = float(dataset[dataset["Locality"] == locality]["Price_INR"].mean()) if locality in dataset["Locality"].values else 0.0
        rows.append({"lat": lat, "lon": lon, "Locality": locality, "Average Price": avg_price})
    return pd.DataFrame(rows)


@st.cache_data
def get_locality_property_trends(locality: str):
    locality_data = dataset[dataset["Locality"] == locality].copy()
    if locality_data.empty:
        return pd.DataFrame({"Property_Type": [], "avg_price": []})
    trend_df = (
        locality_data.groupby("Property_Type", as_index=False)["Price_INR"]
        .mean()
        .rename(columns={"Price_INR": "avg_price"})
        .sort_values("avg_price", ascending=False)
        .reset_index(drop=True)
    )
    trend_df["avg_price"] = trend_df["avg_price"].round(0)
    return trend_df


@st.cache_data
def get_locality_price_trend(locality: str):
    if locality not in dataset["Locality"].unique():
        return pd.DataFrame({"Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"], "Estimated_Price": [0, 0, 0, 0, 0, 0]})

    locality_data = dataset[dataset["Locality"] == locality].copy()
    avg_price = float(locality_data["Price_INR"].mean()) if not locality_data.empty else 0.0
    if avg_price <= 0:
        avg_price = 1500000

    seed_value = sum(ord(ch) for ch in locality) % 7
    trend_adjustments = [0.94, 0.97, 1.00, 1.03, 1.06, 1.09]
    trend_adjustments = [value + (seed_value * 0.01) for value in trend_adjustments]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    return pd.DataFrame(
        {
            "Month": months,
            "Estimated_Price": [round(avg_price * factor, 0) for factor in trend_adjustments],
        }
    )


@st.cache_data
def get_demand_supply_score(locality: str):
    if locality not in dataset["Locality"].unique():
        return {
            "score": 50,
            "label": "Balanced",
            "reason": "No reliable locality demand data available.",
            "listings": 0,
            "avg_price": 0,
        }

    locality_data = dataset[dataset["Locality"] == locality].copy()
    listings = len(locality_data)
    avg_price = float(locality_data["Price_INR"].mean()) if not locality_data.empty else 0.0
    supply_pressure = min(30, max(0, listings / 10))
    demand_strength = min(35, max(0, (avg_price / 500000) * 10))
    quality_bonus = min(25, max(0, get_locality_quality_score(locality)["score"] / 4))
    score = round(min(100, 50 + demand_strength + quality_bonus - supply_pressure))

    if score >= 75:
        label = "Strong demand"
        reason = "Market activity is high, pricing is attractive, and the locality is performing well."
    elif score >= 60:
        label = "Balanced market"
        reason = "Demand and supply are relatively aligned, with moderate investment potential."
    elif score >= 45:
        label = "Watch closely"
        reason = "The locality may need better value positioning or stronger amenities to improve demand."
    else:
        label = "Low momentum"
        reason = "Market interest is limited, and buyers may need stronger pricing or better locality features."

    return {"score": score, "label": label, "reason": reason, "listings": listings, "avg_price": avg_price}


@st.cache_data
def get_property_recommendations(locality: str, property_type: str = None, target_budget: float = None):
    locality_data = dataset[dataset["Locality"] == locality].copy()
    if locality_data.empty:
        return pd.DataFrame({"Property_Type": [], "BHK": [], "Builtup_Area_sqft": [], "Price_INR": [], "Furnishing": [], "Amenities_Level": []})
    if property_type:
        locality_data = locality_data[locality_data["Property_Type"] == property_type]
    if target_budget is not None:
        locality_data = locality_data.copy()
        locality_data["budget_gap"] = (locality_data["Price_INR"] - target_budget).abs()
        locality_data = locality_data.sort_values("budget_gap").head(5)
    else:
        locality_data = locality_data.sort_values("Price_INR").head(5)
    return locality_data[["Property_Type", "BHK", "Builtup_Area_sqft", "Price_INR", "Furnishing", "Amenities_Level"]].reset_index(drop=True)


def normalize_image_url(value) -> str:
    if value is None:
        return PROPERTY_IMAGE_LIBRARY["Default"]
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else PROPERTY_IMAGE_LIBRARY["Default"]
    if isinstance(value, (int, float)) and pd.notna(value):
        return PROPERTY_IMAGE_LIBRARY["Default"]
    if pd.isna(value):
        return PROPERTY_IMAGE_LIBRARY["Default"]
    return PROPERTY_IMAGE_LIBRARY["Default"]


def get_property_image(property_type: str, locality: str) -> str:
    type_key = property_type.strip()
    if type_key in PROPERTY_IMAGE_LIBRARY:
        return normalize_image_url(PROPERTY_IMAGE_LIBRARY[type_key])
    for key, image in PROPERTY_IMAGE_LIBRARY.items():
        if key.lower() in property_type.lower():
            return normalize_image_url(image)
    return normalize_image_url(PROPERTY_IMAGE_LIBRARY["Default"])


def save_prediction(row: dict):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO property_history (
            username, locality, property_type, bhk, bathrooms, builtup_area_sqft, floor,
            property_age_years, furnishing, sale_status, amenities_level, construction_type,
            parking, gated_community, school_distance_km, hospital_distance_km,
            airport_distance_km, railway_distance_km, bus_stand_distance_km,
            main_road_distance_km, pollution_aqi, estimated_price_inr, image_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["username"],
            row["locality"],
            row["property_type"],
            row["bhk"],
            row["bathrooms"],
            row["builtup_area_sqft"],
            row["floor"],
            row["property_age_years"],
            row["furnishing"],
            row["sale_status"],
            row["amenities_level"],
            row["construction_type"],
            row["parking"],
            row["gated_community"],
            row["school_distance_km"],
            row["hospital_distance_km"],
            row["airport_distance_km"],
            row["railway_distance_km"],
            row["bus_stand_distance_km"],
            row["main_road_distance_km"],
            row["pollution_aqi"],
            row["estimated_price_inr"],
            row.get("image_url", PROPERTY_IMAGE_LIBRARY["Default"]),
        ),
    )
    conn.commit()
    conn.close()


def fetch_user_history(username: str):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT * FROM property_history
        WHERE username = ?
        ORDER BY created_at DESC
        LIMIT 20
        """,
        (username,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_investment_guidance(prediction: float, locality: str, property_type: str, amenities_level: str, gated_community: bool):
    insights = get_market_insights()
    locality_data = insights["locality_summary"].loc[insights["locality_summary"]["Locality"] == locality]
    if locality_data.empty:
        locality_average = prediction
    else:
        locality_average = float(locality_data.iloc[0]["avg_price"])

    property_average = float(
        insights["property_summary"].loc[insights["property_summary"]["Property_Type"] == property_type, "avg_price"].iloc[0]
    ) if property_type in insights["property_summary"]["Property_Type"].values else prediction

    agreement = (prediction / locality_average) if locality_average else 1.0
    score = 50
    if agreement > 1.05:
        score += 15
    elif agreement < 0.95:
        score -= 10
    if gated_community:
        score += 10
    if amenities_level in ["Premium", "Standard"]:
        score += 10
    if property_average > 0 and prediction > property_average:
        score += 5
    score = max(0, min(100, score))

    if score >= 80:
        verdict = "Strong investment opportunity"
        reason = "This property is aligned with the premium local market and strong amenity profile."
    elif score >= 60:
        verdict = "Healthy value range"
        reason = "The estimate is competitive and suitable for a balanced investment decision."
    else:
        verdict = "Needs negotiation or improvement"
        reason = "The current estimate is below nearby market expectations, so pricing or upgrades may be needed."

    return score, verdict, reason


def get_ai_buyer_tips(locality: str, property_type: str, bhk: int, bathrooms: int, builtup_area_sqft: float,
                     property_age_years: int, furnishing: str, sale_status: str, amenities_level: str,
                     construction_type: str, gated_community: bool, school_distance_km: float,
                     hospital_distance_km: float, pollution_aqi: int):
    buyer_profiles = {
        "Family Buyer": 0,
        "Investor": 0,
        "Rental Buyer": 0,
    }

    if school_distance_km <= 2.0:
        buyer_profiles["Family Buyer"] += 20
    if hospital_distance_km <= 3.0:
        buyer_profiles["Family Buyer"] += 15
    if gated_community:
        buyer_profiles["Family Buyer"] += 10
        buyer_profiles["Investor"] += 8
    if amenities_level in ["Premium", "Standard"]:
        buyer_profiles["Investor"] += 15
        buyer_profiles["Rental Buyer"] += 10
    if furnishing == "Fully Furnished":
        buyer_profiles["Rental Buyer"] += 15
    if sale_status == "Ready to Move":
        buyer_profiles["Family Buyer"] += 10
        buyer_profiles["Rental Buyer"] += 5
    if property_age_years <= 5:
        buyer_profiles["Investor"] += 10
    if property_age_years >= 15:
        buyer_profiles["Family Buyer"] -= 5
        buyer_profiles["Investor"] -= 5
    if pollution_aqi > 120:
        buyer_profiles["Family Buyer"] -= 10

    best_profile = max(buyer_profiles, key=buyer_profiles.get)
    tips = [
        f"This property is best suited for a {best_profile.lower()} profile based on the current locality and feature mix.",
        f"{locality} is favorable for {property_type.lower()} purchases because the location pattern supports a balanced market demand.",
    ]

    if gated_community:
        tips.append("Gated security adds strong value and improves resale appeal for family buyers.")
    if furnishing == "Fully Furnished":
        tips.append("Fully furnished homes usually attract fast-moving buyers and better rental demand.")
    if property_age_years > 10:
        tips.append("The property is older, so renovation planning or price negotiation may improve the deal value.")
    if school_distance_km <= 2.0 and hospital_distance_km <= 3.0:
        tips.append("The location near schools and hospitals makes this residence attractive for families and long-term tenants.")
    if pollution_aqi > 120:
        tips.append("Air quality is relatively poor in this area, so buyers may prefer a better ventilated or more premium location for long-term value.")
    if amenities_level == "Premium":
        tips.append("Premium amenity levels support stronger resale value and better investor appeal.")

    return best_profile, tips


def calculate_emi(principal: float, annual_rate_percent: float, years: int):
    if principal <= 0 or years <= 0:
        return 0.0, 0.0, 0.0
    monthly_rate = annual_rate_percent / 12 / 100
    months = years * 12
    if monthly_rate == 0:
        emi = principal / months
    else:
        emi = principal * monthly_rate * ((1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)
    total_payable = emi * months
    total_interest = total_payable - principal
    return emi, total_interest, total_payable


def calculate_roi(purchase_price: float, monthly_rent: float, annual_maintenance: float = 0.0, vacancy_rate: float = 0.0):
    if purchase_price <= 0:
        return 0.0, 0.0, 0.0, 0.0
    annual_rent = monthly_rent * 12
    effective_income = annual_rent * (1 - vacancy_rate / 100.0)
    net_income = effective_income - annual_maintenance
    gross_yield = (annual_rent / purchase_price) * 100
    net_yield = (net_income / purchase_price) * 100
    annual_profit = net_income
    return gross_yield, net_yield, annual_profit, effective_income


@st.cache_data
def get_locality_quality_score(locality: str):
    if locality not in dataset["Locality"].unique():
        return {"score": 50, "label": "Moderate", "reason": "This locality is not in the trained dataset."}

    locality_data = dataset[dataset["Locality"] == locality].copy()
    avg_price = float(locality_data["Price_INR"].mean()) if not locality_data.empty else 0.0
    avg_school = float(locality_data["School_Distance_km"].mean()) if "School_Distance_km" in locality_data.columns else 3.0
    avg_hospital = float(locality_data["Hospital_Distance_km"].mean()) if "Hospital_Distance_km" in locality_data.columns else 4.0
    avg_pollution = float(locality_data["Pollution_AQI_Approx"].mean()) if "Pollution_AQI_Approx" in locality_data.columns else 100
    amenity_score = 50
    if avg_school <= 2.5:
        amenity_score += 15
    if avg_hospital <= 3.5:
        amenity_score += 15
    if avg_pollution <= 110:
        amenity_score += 10
    if avg_price > 0:
        amenity_score += min(20, max(0, (avg_price / 1000000) * 10))

    score = max(0, min(100, round(amenity_score)))
    if score >= 80:
        label = "Excellent"
        reason = "Strong local demand, good access, and balanced lifestyle factors."
    elif score >= 65:
        label = "Good"
        reason = "Healthy value profile with practical amenities and decent connectivity."
    elif score >= 45:
        label = "Moderate"
        reason = "Balanced but may need better amenities or better transport access."
    else:
        label = "Budget"
        reason = "Lower convenience profile and weaker market attractiveness."

    return {"score": score, "label": label, "reason": reason, "avg_price": avg_price}


def add_to_watchlist(property_data: dict):
    watchlist = st.session_state.get("watchlist", [])
    key = (
        property_data.get("locality"),
        property_data.get("property_type"),
        property_data.get("estimated_price_inr"),
        property_data.get("bhk"),
    )
    if any(
        item.get("locality") == key[0]
        and item.get("property_type") == key[1]
        and float(item.get("estimated_price_inr", 0)) == float(key[2])
        and item.get("bhk") == key[3]
        for item in watchlist
    ):
        return False
    watchlist.append(property_data)
    st.session_state["watchlist"] = watchlist
    return True


def get_watchlist_summary():
    watchlist = st.session_state.get("watchlist", [])
    if not watchlist:
        return pd.DataFrame(columns=["locality", "property_type", "bhk", "estimated_price_inr", "price_per_sqft"])

    summary = []
    for item in watchlist:
        area = float(item.get("builtup_area_sqft", 0) or 0)
        price = float(item.get("estimated_price_inr", 0) or 0)
        summary.append(
            {
                "locality": item.get("locality"),
                "property_type": item.get("property_type"),
                "bhk": item.get("bhk"),
                "estimated_price_inr": price,
                "price_per_sqft": round(price / area, 2) if area > 0 else 0,
            }
        )
    return pd.DataFrame(summary)


def get_locality_amenity_markers(locality: str):
    if locality not in LOCALITY_COORDS:
        return []

    lat, lon = LOCALITY_COORDS[locality]
    amenities = [
        {"name": "School", "type": "School", "distance_km": 1.4, "angle_deg": 35, "color": "#22c55e"},
        {"name": "Hospital", "type": "Hospital", "distance_km": 2.2, "angle_deg": 145, "color": "#ef4444"},
        {"name": "Shops & Market", "type": "Shops", "distance_km": 1.7, "angle_deg": 280, "color": "#f59e0b"},
        {"name": "Bus Stop", "type": "Transit", "distance_km": 0.9, "angle_deg": 85, "color": "#3b82f6"},
        {"name": "Park", "type": "Park", "distance_km": 1.2, "angle_deg": 220, "color": "#10b981"},
        {"name": "Railway", "type": "Transit", "distance_km": 2.8, "angle_deg": 330, "color": "#8b5cf6"},
    ]

    points = []
    for amenity in amenities:
        angle_rad = math.radians(amenity["angle_deg"])
        lat_delta = amenity["distance_km"] * math.cos(angle_rad) / 111.32
        lon_delta = amenity["distance_km"] * math.sin(angle_rad) / (111.32 * math.cos(math.radians(lat)))
        points.append({
            "name": f"{amenity['name']} near {locality}",
            "type": amenity["type"],
            "lat": lat + lat_delta,
            "lon": lon + lon_delta,
            "color": amenity["color"],
        })

    return points


def render_locality_map_selector():
    selected = st.session_state.get("selected_map_locality", LOCALITIES[0] if LOCALITIES else "")
    markers = []
    for locality, (lat, lon) in LOCALITY_COORDS.items():
        markers.append({
            "name": locality,
            "lat": lat,
            "lon": lon,
            "selected": locality == selected,
        })

    selected_json = json.dumps(selected)
    marker_json = json.dumps(markers)
    html = (
        """
        <div id="real-estate-map" style="height: 420px; width: 100%; border-radius: 14px; overflow: hidden;"></div>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
          const selectedLocality = """
        + selected_json
        + """;
          const markers = """
        + marker_json
        + """;

          const map = L.map('real-estate-map', { zoomControl: true }).setView([19.876, 75.34], 11.6);
          L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
          }).addTo(map);

          const makeIcon = (isSelected) => {
            return L.divIcon({
              className: 'custom-marker',
              html: `<div style="background:${isSelected ? '#22c55e' : '#2563eb'}; width: 18px; height: 18px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.15);"></div>`,
              iconSize: [18, 18],
              iconAnchor: [9, 9]
            });
          };

          let selectedMarkerLayer = null;
          const updateSelection = (name) => {
            if (!name) return;
            const payload = { type: 'streamlit:setComponentValue', value: name };
            window.parent.postMessage(payload, '*');
            const current = window.parent;
            if (current && current.__streamlitMessageListener) {
              current.__streamlitMessageListener({ type: 'location-selection', value: name });
            }
          };

          markers.forEach((markerInfo) => {
            const marker = L.marker([markerInfo.lat, markerInfo.lon], { icon: makeIcon(markerInfo.selected) }).addTo(map);
            marker.bindPopup(markerInfo.name);
            marker.on('click', () => {
              updateSelection(markerInfo.name);
              if (selectedMarkerLayer) {
                selectedMarkerLayer.setIcon(makeIcon(false));
              }
              selectedMarkerLayer = marker;
              marker.setIcon(makeIcon(true));
            });
          });

          if (selectedLocality) {
            const selectedMarker = markers.find((m) => m.name === selectedLocality);
            if (selectedMarker) {
              map.flyTo([selectedMarker.lat, selectedMarker.lon], 12.5, { duration: 0.7 });
            }
          }
        </script>
        """
    )
    return components.html(html, height=430, scrolling=False)


def render_locality_amenity_map(locality: str):
    if locality not in LOCALITY_COORDS:
        st.info("Locality map is not available for this selection.")
        return

    lat, lon = LOCALITY_COORDS[locality]
    amenities = get_locality_amenity_markers(locality)
    locality_json = json.dumps({"name": locality, "lat": lat, "lon": lon})
    amenity_json = json.dumps(amenities)

    html = (
        """
        <div id="locality-amenities-map" style="height: 340px; width: 100%; border-radius: 14px; overflow: hidden;"></div>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
          const locality = """
        + locality_json
        + """;
          const amenities = """
        + amenity_json
        + """;

          const map = L.map('locality-amenities-map', { zoomControl: true }).setView([locality.lat, locality.lon], 14);
          L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors'
          }).addTo(map);

          const localityCircle = L.circleMarker([locality.lat, locality.lon], {
            radius: 13,
            color: '#f59e0b',
            fillColor: '#f59e0b',
            fillOpacity: 0.9,
            weight: 2
          }).addTo(map);
          localityCircle.bindPopup(`<strong>${locality}</strong><br/>Selected property locality`);

          amenities.forEach((item) => {
            const marker = L.circleMarker([item.lat, item.lon], {
              radius: 8,
              color: item.color,
              fillColor: item.color,
              fillOpacity: 0.9,
              weight: 2
            }).addTo(map);
            marker.bindPopup(`<strong>${item.type}</strong><br/>${item.name}`);
          });

          const allPoints = amenities.map((item) => [item.lat, item.lon]);
          allPoints.push([locality.lat, locality.lon]);
          if (allPoints.length > 1) {
            const bounds = L.latLngBounds(allPoints);
            map.fitBounds(bounds.pad(0.45));
          }
        </script>
        """
    )
    return components.html(html, height=360, scrolling=False)


def get_buy_vs_rent_recommendation(property_price: float, down_payment: float, annual_interest_rate: float,
                                  loan_years: int, monthly_rent: float, annual_maintenance: float,
                                  expected_stay_years: int, annual_appreciation: float = 5.0):
    if property_price <= 0 or expected_stay_years <= 0:
        return "Rent for now", "Insufficient data to decide confidently.", 0.0, 0.0

    loan_amount = max(property_price - down_payment, 0)
    emi, _, _ = calculate_emi(loan_amount, annual_interest_rate, loan_years)
    total_buy_cost = emi + (annual_maintenance / 12)
    buy_score = 50
    if total_buy_cost < monthly_rent:
        buy_score += 20
    if monthly_rent <= 0:
        buy_score += 10

    expected_value = property_price * ((1 + annual_appreciation / 100) ** expected_stay_years)
    appreciation_gain = expected_value - property_price
    buy_score += min(20, max(0, appreciation_gain / 100000))

    if total_buy_cost <= monthly_rent and buy_score >= 70:
        verdict = "Buy"
        reason = "Monthly cost is reasonable and long-term appreciation makes buying more favorable."
    elif total_buy_cost > monthly_rent and buy_score < 60:
        verdict = "Rent for now"
        reason = "Renting is financially safer unless you expect a stronger local market rise."
    else:
        verdict = "Buy or wait"
        reason = "The choice depends on your budget and how long you plan to stay in the property."

    return verdict, reason, total_buy_cost, expected_value


init_db()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = []
if "selected_map_locality" not in st.session_state:
    st.session_state["selected_map_locality"] = ""

st.set_page_config(page_title="Real Estate Price Prediction System Chh. Sambhajinagar", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(2, 6, 23, 0.62), rgba(2, 6, 23, 0.72)),
            url("https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=1600&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main {
        background: transparent;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .auth-brand {
        display: none;
    }
    .sidebar-logo-card {
        background: linear-gradient(180deg, #071a32 0%, #050f1f 100%);
        border: 1px solid rgba(212, 175, 55, 0.28);
        border-radius: 18px;
        padding: 0.8rem 0.7rem 0.9rem;
        box-shadow: 0 18px 30px rgba(2, 6, 23, 0.32);
        margin-bottom: 0.9rem;
        position: relative;
        overflow: hidden;
    }
    .sidebar-logo-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.06), rgba(255,255,255,0.02), rgba(212, 175, 55, 0.04));
        pointer-events: none;
    }
    .csn-logo-wrap {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.2rem;
        width: 180px;
        height: 110px;
        z-index: 1;
    }
    .csn-logo-mark {
        position: relative;
        z-index: 1;
        width: 150px;
        height: 90px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: transparent;
        border: none;
        border-radius: 0;
        box-shadow: none;
    }
    .csn-logo-mark svg {
        width: 150px;
        height: 88px;
        display: block;
        filter: drop-shadow(0 8px 14px rgba(212, 175, 55, 0.18));
    }
    .csn-brand-text {
        text-align: center;
        line-height: 1.05;
        position: relative;
        z-index: 1;
    }
    .csn-brand-tag {
        display: block;
        margin-bottom: 0.15rem;
        font-size: 0.58rem;
        font-weight: 800;
        letter-spacing: 0.34em;
        color: #d4af37;
        text-transform: uppercase;
    }
    .csn-brand-name {
        display: block;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        color: #e7d39a;
        text-transform: uppercase;
    }
    .csn-brand-sub {
        display: block;
        margin-top: 0.18rem;
        font-size: 0.72rem;
        letter-spacing: 0.38em;
        text-transform: uppercase;
        color: #d4af37;
        font-weight: 700;
    }
    .csn-logo-header {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0.1rem auto 0.7rem;
        width: 100%;
    }
    .csn-logo-header .csn-logo-badge {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #d4af37;
        width: min(700px, 72vw);
        background: rgba(5, 18, 39, 0.8);
        border-radius: 22px;
        padding: 1rem 1rem 0.9rem;
        box-shadow: 0 18px 40px rgba(2, 6, 23, 0.26);
    }
    .csn-logo-header svg {
        width: min(440px, 62vw);
        height: auto;
        display: block;
        filter: drop-shadow(0 12px 20px rgba(212, 175, 55, 0.16));
    }
    .csn-logo-header .csn-wordmark {
        font-size: clamp(4rem, 8vw, 7rem);
        font-weight: 900;
        letter-spacing: 0.02em;
        line-height: 0.9;
        color: #d4af37;
        text-transform: uppercase;
        margin-top: -0.5rem;
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
    }
    .csn-logo-header .csn-subword {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        width: 100%;
        margin-top: 0.2rem;
        color: #e4c96d;
        font-size: clamp(0.85rem, 1.75vw, 1.65rem);
        letter-spacing: 0.28em;
        text-transform: uppercase;
        font-weight: 700;
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
    }
    .csn-logo-header .csn-subword::before,
    .csn-logo-header .csn-subword::after {
        content: "";
        display: inline-block;
        width: clamp(32px, 7vw, 110px);
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(212,175,55,0.9), transparent);
    }
    .hero {
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(6px);
        color: white;
        border-radius: 24px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    .hero h1 {
        margin: 0.2rem 0 0.5rem 0;
        font-size: 2.4rem;
        font-weight: 800;
    }
    .hero p {
        margin: 0;
        color: #dbeafe;
        font-size: 1rem;
    }
    .small-label {
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #a5b4fc;
        font-weight: 700;
    }
    .card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.24);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
        border-radius: 22px;
        padding: 1.4rem 1.2rem;
        color: white;
        box-shadow: 0 18px 40px rgba(249, 115, 22, 0.28);
        border: 1px solid rgba(255,255,255,0.2);
        overflow: hidden;
        position: relative;
        animation: priceGlow 3s ease-in-out infinite alternate;
    }
    .prediction-box::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, rgba(255,255,255,0.12), rgba(255,255,255,0.02), rgba(255,255,255,0.12));
        transform: translateX(-100%);
        animation: sheen 4s ease-in-out infinite;
    }
    .prediction-box h3 {
        margin: 0 0 0.25rem 0;
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.95;
    }
    .prediction-box .price {
        font-size: 3rem;
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -0.04em;
        text-shadow: 0 6px 24px rgba(15, 23, 42, 0.25);
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
        animation: floatPrice 2.4s ease-in-out infinite alternate;
    }
    .premium-hero {
        background: linear-gradient(135deg, #0f172a 0%, #111827 38%, #1d4ed8 100%);
        border-radius: 28px;
        padding: 1.5rem 1.6rem;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 22px 46px rgba(59, 130, 246, 0.22);
        border: 1px solid rgba(96, 165, 250, 0.18);
        animation: glowPulse 4s ease-in-out infinite alternate;
    }
    .premium-hero h1 {
        margin: 0 0 0.4rem 0;
        font-size: 2.7rem;
        font-weight: 900;
    }
    .premium-hero p {
        margin: 0;
        color: #dbeafe;
        font-size: 1rem;
    }
    .feature-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin: 1.2rem 0 1.5rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.9));
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 18px 34px rgba(15,23,42,0.22);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        animation: floatCard 5s ease-in-out infinite alternate;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 38px rgba(59,130,246,0.15);
        border-color: rgba(96,165,250,0.4);
    }
    .feature-card .icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        background: linear-gradient(135deg, #1d4ed8 0%, #a78bfa 100%);
        margin-bottom: 0.6rem;
    }
    .feature-card h4 {
        margin: 0.15rem 0 0.25rem 0;
        color: #f8fafc;
        font-size: 1.02rem;
    }
    .feature-card p {
        margin: 0;
        color: #cbd5e1;
        font-size: 0.88rem;
    }
    .gallery-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .gallery-item {
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 22px;
        overflow: hidden;
        box-shadow: 0 16px 32px rgba(15,23,42,0.22);
        transition: transform 0.25s ease;
    }
    .gallery-item:hover {
        transform: translateY(-2px);
    }
    .gallery-item img {
        width: 100%;
        height: 190px;
        object-fit: cover;
        display: block;
    }
    .gallery-body {
        padding: 0.9rem 1rem 1rem 1rem;
    }
    .gallery-body h4 {
        margin: 0 0 0.35rem 0;
        color: #f8fafc;
    }
    .gallery-body p {
        margin: 0.15rem 0;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    .metric-chip {
        display: inline-block;
        background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%);
        color: #f8fafc;
        font-weight: 700;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.72rem;
        margin-top: 0.55rem;
    }
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        height: 3rem;
        border: none;
        background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
        color: white;
        border-radius: 12px;
        font-weight: 700;
        box-shadow: 0 12px 24px rgba(249,115,22,0.25);
    }
    @keyframes glowPulse {
        from { box-shadow: 0 18px 38px rgba(59,130,246,0.12); }
        to { box-shadow: 0 24px 52px rgba(96,165,250,0.28); }
    }
    @keyframes logoFloat {
        from { transform: translateY(0px) scale(1); }
        to { transform: translateY(-4px) scale(1.05); }
    }
    @keyframes logoOrbit {
        0% { transform: rotate(0deg) scale(1); }
        50% { transform: rotate(180deg) scale(1.08); }
        100% { transform: rotate(360deg) scale(1); }
    }
    @keyframes floatCard {
        from { transform: translateY(0px); }
        to { transform: translateY(-2px); }
    }
    @keyframes priceGlow {
        from { box-shadow: 0 18px 40px rgba(249, 115, 22, 0.22); }
        to { box-shadow: 0 24px 52px rgba(249, 115, 22, 0.38); }
    }
    @keyframes floatPrice {
        from { transform: translateY(0px); }
        to { transform: translateY(-3px); }
    }
    @keyframes sheen {
        0% { transform: translateX(-100%); }
        40% { transform: translateX(100%); }
        100% { transform: translateX(100%); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.session_state["authenticated"]:
    st.markdown(
        """
        <div class="hero">
            <div class="small-label">Secure Access</div>
            <h1>Real Estate Price Prediction System Chh. Sambhajinagar</h1>
            <p>Login to access property valuation, neighborhood insights, and saved property history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    auth_tab = st.tabs(["Login", "Register"])
    with auth_tab[0]:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            conn = get_db_connection()
            record = conn.execute(
                "SELECT username, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if record and verify_password(password, record["password_hash"]):
                if not record["password_hash"].startswith(f"{PASSWORD_HASH_PREFIX}$"):
                    conn.execute(
                        "UPDATE users SET password_hash = ? WHERE username = ?",
                        (hash_password(password), username),
                    )
                    conn.commit()
                conn.close()
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success("Login successful.")
                st.rerun()
            else:
                conn.close()
                st.error("Invalid username or password.")

    with auth_tab[1]:
        new_username = st.text_input("New username", key="new_username")
        full_name = st.text_input("Full name", key="full_name")
        new_password = st.text_input("Create password", type="password", key="new_password")
        if st.button("Create Account"):
            if not new_username or not full_name or not new_password:
                st.warning("Please fill in all fields.")
            else:
                conn = get_db_connection()
                existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (new_username,)).fetchone()
                if existing:
                    st.error("Username already exists.")
                else:
                    conn.execute(
                        "INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
                        (new_username, hash_password(new_password), full_name),
                    )
                    conn.commit()
                    conn.close()
                    st.success("Account created successfully. Please log in.")

    st.stop()

st.markdown(
    """
    <div class="premium-hero">
        <div class="small-label">Smart Market Intelligence</div>
        <h1>Chhatrapati Sambhajinagar Property Pricing</h1>
        <p>AI-powered valuation dashboard for homes, flats, villas, and investment planning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="feature-strip">
        <div class="feature-card">
            <div class="icon">🏠</div>
            <h4>Smart Pricing</h4>
            <p>Data-led property valuation for real local conditions.</p>
        </div>
        <div class="feature-card">
            <div class="icon">📍</div>
            <h4>Local Market Insights</h4>
            <p>Compare locality trends and neighborhood opportunities.</p>
        </div>
        <div class="feature-card">
            <div class="icon">💬</div>
            <h4>Buyer Guidance</h4>
            <p>Get recommendations tailored to your property profile.</p>
        </div>
        <div class="feature-card">
            <div class="icon">📄</div>
            <h4>Instant Reports</h4>
            <p>Save estimates and export project-ready reports in one click.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="gallery-grid">
        <div class="gallery-item">
            <img src="https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?auto=format&fit=crop&w=900&q=80" alt="Luxury apartment" />
            <div class="gallery-body">
                <h4>Signature Apartments</h4>
                <p>3 BHK | Premium skyline view</p>
                <p>₹68 Lakh</p>
                <span class="metric-chip">Luxury</span>
            </div>
        </div>
        <div class="gallery-item">
            <img src="https://images.unsplash.com/photo-1494526585095-c41746248156?auto=format&fit=crop&w=900&q=80" alt="Modern villa" />
            <div class="gallery-body">
                <h4>Modern Villa</h4>
                <p>4 BHK | Garden frontage</p>
                <p>₹1.4 Cr</p>
                <span class="metric-chip">High Demand</span>
            </div>
        </div>
        <div class="gallery-item">
            <img src="https://images.unsplash.com/photo-1484154218962-a197022b5858?auto=format&fit=crop&w=900&q=80" alt="Family home" />
            <div class="gallery-body">
                <h4>Family Residence</h4>
                <p>2 BHK | Smart community</p>
                <p>₹42 Lakh</p>
                <span class="metric-chip">Best Value</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.rerun()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo-card">
            <div class="csn-logo-wrap">
                <div class="csn-logo-mark">
                    <svg viewBox="0 0 300 170" xmlns="http://www.w3.org/2000/svg" aria-label="CSN Real Estate logo">
                        <defs>
                            <linearGradient id="csnGoldSidebar" x1="0%" x2="100%" y1="0%" y2="100%">
                                <stop offset="0%" stop-color="#f8e7b6"/>
                                <stop offset="50%" stop-color="#d4af37"/>
                                <stop offset="100%" stop-color="#b88a2b"/>
                            </linearGradient>
                        </defs>
                        <g fill="none" stroke="url(#csnGoldSidebar)" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M52 130V74h24V52h25v22h25v56"/>
                            <path d="M28 130V90h18v40"/>
                            <path d="M172 130V52h24V34h24v18h25v78"/>
                            <path d="M104 130V96h18v34"/>
                            <path d="M220 130V82h20v48"/>
                            <path d="M88 130H248"/>
                            <path d="M65 108h12M65 94h12M65 80h12M118 108h12M118 94h12M118 80h12M191 108h12M191 94h12M191 80h12M242 108h12M242 94h12M242 80h12"/>
                        </g>
                    </svg>
                </div>
            </div>
            <div class="csn-brand-text">
                <span class="csn-brand-name">REAL ESTATE</span>
                <span class="csn-brand-sub">PROPERTY</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    market_insights = get_market_insights()
    st.markdown("---")
    st.subheader("Market Snapshot")

    selected_locality = st.session_state.get("selected_map_locality", LOCALITIES[0] if LOCALITIES else "")
    selected_locality = str(selected_locality) if selected_locality is not None else ""
    snapshot_df = dataset.copy()
    selected_market = (
        snapshot_df.loc[snapshot_df["Locality"].astype(str).eq(selected_locality)].copy()
        if selected_locality
        else pd.DataFrame()
    )
    selected_avg = float(selected_market["Price_INR"].mean()) if not selected_market.empty else 0.0
    trend_label = "Strong demand" if selected_avg >= market_insights["premium_locality"]["avg_price"] * 0.9 else "Balanced"

    st.caption(f"Latest update: {pd.Timestamp.now().strftime('%d %b %Y, %H:%M')}")
    st.metric("Selected locality", selected_locality or "N/A", f"₹{selected_avg:,.0f}")
    st.metric("Top locality", market_insights["premium_locality"]["Locality"], f"₹{market_insights['premium_locality']['avg_price']:,.0f}")
    st.metric("Budget area", market_insights["affordable_locality"]["Locality"], f"₹{market_insights['affordable_locality']['avg_price']:,.0f}")
    st.metric("Best property type", market_insights["best_property_type"]["Property_Type"], f"₹{market_insights['best_property_type']['avg_price']:,.0f}")
    st.caption(f"Market pulse: {trend_label}")

if st.session_state["authenticated"]:
    tabs = st.tabs(["Property Valuation", "Neighborhood Map", "Saved History", "Decision Support", "Market Analytics", "Reports"])
    property_report_csv = None

    with tabs[0]:
        with st.form("prediction_form"):
            st.subheader("Property Search")
            search_col, filter_col = st.columns([2, 1])

            with search_col:
                selected_map_locality = st.session_state.get("selected_map_locality", DEFAULT_LOCALITY)
                locality_value = selected_map_locality if selected_map_locality in LOCALITIES else "Select a locality"
                default_locality_index = LOCALITY_OPTIONS.index(locality_value) if locality_value in LOCALITY_OPTIONS else 0
                locality = st.selectbox("Locality", LOCALITY_OPTIONS, index=default_locality_index, key="property_locality_select")
                if locality != "Select a locality":
                    st.session_state["selected_map_locality"] = locality
                    st.session_state["selected_locality_map"] = locality
                else:
                    st.session_state["selected_map_locality"] = ""
                st.caption(f"Selected Location: {locality if locality != 'Select a locality' else 'Not selected'}")
            with filter_col:
                property_type = st.selectbox("Property Type", PROPERTY_TYPES, index=0)

            st.caption("Fast property search using clean filters and dropdown inputs.")
            st.markdown("---")

            st.markdown("### Basic property details")
            col1, col2 = st.columns(2)

            with col1:
                bhk = st.slider("BHK", min_value=0, max_value=5, value=2)
                bathrooms = st.slider("Bathrooms", min_value=0, max_value=5, value=2)
                builtup_area_sqft = st.number_input("Built-up Area (sq ft)", min_value=300, max_value=5000, value=1200)
                floor = st.slider("Floor", min_value=0, max_value=20, value=4)
                property_age_years = st.slider("Property Age (years)", min_value=0, max_value=30, value=8)
                furnishing = st.selectbox("Furnishing", FURNISHING_OPTIONS)

            with col2:
                sale_status = st.selectbox("Sale Status", SALE_STATUS_OPTIONS)
                amenities_level = st.selectbox("Amenities Level", AMENITY_LEVELS)
                construction_type = st.selectbox("Construction Type", CONSTRUCTION_TYPES)
                parking = st.slider("Parking", min_value=0, max_value=2, value=1)
                gated_community = st.checkbox("Gated Community")

            st.markdown("### Location & connectivity")
            loc_col1, loc_col2 = st.columns(2)

            with loc_col1:
                school_distance = st.slider("School Distance (km)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
                hospital_distance = st.slider("Hospital Distance (km)", min_value=0.5, max_value=12.0, value=3.0, step=0.1)
                airport_distance = st.slider("Airport Distance (km)", min_value=0.5, max_value=25.0, value=8.0, step=0.1)
                railway_distance = st.slider("Railway Distance (km)", min_value=0.5, max_value=15.0, value=4.0, step=0.1)

            with loc_col2:
                bus_distance = st.slider("Bus Stand Distance (km)", min_value=0.2, max_value=10.0, value=2.0, step=0.1)
                main_road_distance = st.slider("Main Road Distance (km)", min_value=0.1, max_value=5.0, value=0.8, step=0.1)
                pollution_aqi = st.slider("Pollution AQI", min_value=80, max_value=150, value=105)

            submitted = st.form_submit_button("Calculate Estimated Price")

            if submitted:
                if locality == "Select a locality":
                    st.warning("Please select a location from the Sambhajinagar map.")
                    st.stop()

                total_rooms = int(bhk if bhk > 0 else 1)
                area_per_bhk = round(builtup_area_sqft / bhk, 1) if bhk > 0 else float(builtup_area_sqft)

                input_data = pd.DataFrame(
                    [{
                        "Locality": locality,
                        "Property_Type": property_type,
                        "BHK": bhk,
                        "Bathrooms": bathrooms,
                        "Builtup_Area_sqft": builtup_area_sqft,
                        "Floor": floor,
                        "Property_Age_Years": property_age_years,
                        "Furnishing": furnishing,
                        "Sale_Status": sale_status,
                        "Amenities_Level": amenities_level,
                        "Construction_Type": construction_type,
                        "Parking": parking,
                        "Gated_Community": int(gated_community),
                        "School_Distance_km": school_distance,
                        "Hospital_Distance_km": hospital_distance,
                        "Airport_Distance_km": airport_distance,
                        "Railway_Distance_km": railway_distance,
                        "Bus_Stand_Distance_km": bus_distance,
                        "Main_Road_Distance_km": main_road_distance,
                        "Pollution_AQI_Approx": pollution_aqi,
                        "Total_Rooms_Approx": total_rooms,
                        "Area_per_BHK_sqft": area_per_bhk,
                    }],
                    columns=FEATURE_COLUMNS,
                )

                prediction = float(model.predict(input_data)[0])
                avg_price, median_price, _ = get_locality_summary(locality)
                lower_range = prediction * 0.94
                upper_range = prediction * 1.08
                predicted_price_per_sqft = prediction / builtup_area_sqft if builtup_area_sqft > 0 else 0
                market_status = "Above locality average" if prediction > avg_price else "Below locality average"

                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h3>Estimated Property Value</h3>
                        <div class="price">₹{prediction:,.0f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                metric_cols = st.columns(4)
                metric_cols[0].metric("Locality average", f"₹{avg_price:,.0f}")
                metric_cols[1].metric("Median value", f"₹{median_price:,.0f}")
                metric_cols[2].metric("Price / sq ft", f"₹{predicted_price_per_sqft:,.0f}")
                metric_cols[3].metric("Market trend", market_status)

                st.markdown("### Nearby surroundings in this locality")
                render_locality_amenity_map(locality)

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="small-label">Valuation range</div>
                        <p><strong>₹{lower_range:,.0f}</strong> to <strong>₹{upper_range:,.0f}</strong></p>
                        <p>This estimate is based on the real dataset for <strong>{locality}</strong> and closely matches the current local sale pattern.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                investment_score, verdict, reason = get_investment_guidance(prediction, locality, property_type, amenities_level, gated_community)
                st.subheader("Investment guidance")
                st.progress(float(investment_score) / 100.0)
                st.write(f"**{verdict}**")
                st.caption(reason)

                best_profile, buyer_tips = get_ai_buyer_tips(
                    locality,
                    property_type,
                    bhk,
                    bathrooms,
                    builtup_area_sqft,
                    property_age_years,
                    furnishing,
                    sale_status,
                    amenities_level,
                    construction_type,
                    gated_community,
                    school_distance,
                    hospital_distance,
                    pollution_aqi,
                )

                with st.container():
                    st.subheader("AI buyer recommendation tips")
                    st.info(f"Recommended buyer profile: {best_profile}")
                    for tip in buyer_tips:
                        st.write("•", tip)

                if gated_community and furnishing == "Fully Furnished":
                    st.success("This property appears premium due to gated security and furnished interiors.")
                elif gated_community:
                    st.info("Gated community status is adding steady value in this locality.")
                elif furnishing == "Fully Furnished":
                    st.info("Furnished properties often receive stronger buyer interest and better resale value.")
                else:
                    st.warning("This property may need better finish, amenities, or security to match the premium local market range.")

                property_market_data = get_market_insights()["property_summary"]
                property_type_average = (
                    float(property_market_data.loc[property_market_data["Property_Type"] == property_type, "avg_price"].iloc[0])
                    if property_type in property_market_data["Property_Type"].values
                    else prediction
                )
                comparison_df = pd.DataFrame(
                    {
                        "Category": ["Your estimate", "Locality average", "Property type average"],
                        "Value": [prediction, avg_price, property_type_average],
                    }
                )
                st.dataframe(comparison_df, width="stretch")

                st.subheader("Best matching properties in this locality")
                recommendations = get_property_recommendations(locality, property_type, prediction)
                if recommendations.empty:
                    st.info("No similar property suggestions available for this locality yet.")
                else:
                    st.dataframe(recommendations, width="stretch")

                property_image = normalize_image_url(get_property_image(property_type, locality))
                history_row = {
                    "username": st.session_state["username"],
                    "locality": locality,
                    "property_type": property_type,
                    "bhk": bhk,
                    "bathrooms": bathrooms,
                    "builtup_area_sqft": builtup_area_sqft,
                    "floor": floor,
                    "property_age_years": property_age_years,
                    "furnishing": furnishing,
                    "sale_status": sale_status,
                    "amenities_level": amenities_level,
                    "construction_type": construction_type,
                    "parking": parking,
                    "gated_community": int(gated_community),
                    "school_distance_km": float(school_distance),
                    "hospital_distance_km": float(hospital_distance),
                    "airport_distance_km": float(airport_distance),
                    "railway_distance_km": float(railway_distance),
                    "bus_stand_distance_km": float(bus_distance),
                    "main_road_distance_km": float(main_road_distance),
                    "pollution_aqi": int(pollution_aqi),
                    "estimated_price_inr": float(prediction),
                    "image_url": property_image,
                }

                st.image(property_image, caption=f"{property_type} in {locality}", width="stretch")
                save_prediction(history_row)
                st.success("Saved to your property history.")

                if st.form_submit_button("Add to Watchlist"):
                    added = add_to_watchlist(history_row)
                    if added:
                        st.success("Property added to your watchlist.")
                    else:
                        st.info("This property is already in your watchlist.")

                report_df = pd.DataFrame([history_row])
                property_report_csv = report_df.to_csv(index=False).encode("utf-8")

        if property_report_csv is not None:
            st.download_button(
                label="Download Property Report",
                data=property_report_csv,
                file_name=f"property_report_{st.session_state['username']}.csv",
                mime="text/csv",
            )

    with tabs[1]:
        st.markdown("### Neighborhood map view")
        nearby_localities = get_neighborhood_map_data()

        selected_map_value = render_locality_map_selector()
        if selected_map_value:
            st.session_state["selected_map_locality"] = selected_map_value
            st.session_state["selected_locality_map"] = selected_map_value
            st.success(f"{selected_map_value} selected from map and auto-filled into the valuation form.")

        selected_locality = st.session_state.get("selected_map_locality", LOCALITIES[0])
        st.info(f"Currently selected locality: {selected_locality}")

        locality_columns = st.columns(3)
        for idx, row in nearby_localities.iterrows():
            with locality_columns[idx % 3]:
                locality_name = row["Locality"]
                is_selected = locality_name == selected_locality
                button_label = f"{'✅' if is_selected else '📍'} {locality_name}"
                if st.button(button_label, key=f"map_select_{locality_name}", type="primary" if is_selected else "secondary"):
                    st.session_state["selected_map_locality"] = locality_name
                    st.session_state["selected_locality_map"] = locality_name
                    st.success(f"{locality_name} selected for valuation.")

        st.markdown("---")
        selected_locality = st.selectbox("Focus on locality", ["All"] + list(LOCALITY_COORDS.keys()), index=list(LOCALITY_COORDS.keys()).index(selected_locality) + 1 if selected_locality in LOCALITY_COORDS else 0)
        if selected_locality == "All":
            st.dataframe(nearby_localities, width="stretch")
        else:
            filtered = nearby_localities[nearby_localities["Locality"] == selected_locality]
            st.dataframe(filtered, width="stretch")

    with tabs[2]:
        history = fetch_user_history(st.session_state["username"])
        if history:
            history_df = pd.DataFrame(history)
            for _, row in history_df.head(6).iterrows():
                col1, col2 = st.columns([1, 2])
                with col1:
                    image_url = normalize_image_url(row.get("image_url"))
                    st.image(image_url, width=220)
                with col2:
                    st.subheader(f"{row['locality']} | {row['property_type']}")
                    st.write(f"BHK: {row['bhk']} | Price: ₹{float(row['estimated_price_inr']):,.0f}")
                    st.write(f"Furnishing: {row['furnishing']} | Status: {row['sale_status']}")
                    st.caption(row.get("created_at", ""))
            st.dataframe(history_df[["locality", "property_type", "bhk", "estimated_price_inr", "created_at"]], width="stretch")
        else:
            st.info("No saved property estimates yet.")

    with tabs[3]:
        st.markdown("### Decision support studio")

        decision_col1, decision_col2 = st.columns(2)
        with decision_col1:
            locality_for_decision = st.selectbox("Locality for decision analysis", LOCALITIES)
            locality_quality = get_locality_quality_score(locality_for_decision)
            st.metric("Locality quality score", f"{locality_quality['score']}/100", locality_quality['label'])
            st.info(locality_quality["reason"])

            property_price = st.number_input("Property purchase price (₹)", min_value=100000, max_value=100000000, value=2500000, step=50000)
            monthly_rent = st.number_input("Expected monthly rent (₹)", min_value=5000, max_value=200000, value=30000, step=1000)
            annual_maintenance = st.number_input("Annual maintenance cost (₹)", min_value=0, max_value=1000000, value=20000, step=1000)
            vacancy_rate = st.slider("Vacancy rate (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)

            gross_yield, net_yield, annual_profit, effective_income = calculate_roi(
                property_price,
                monthly_rent,
                annual_maintenance,
                vacancy_rate,
            )

            st.subheader("Rental ROI")
            st.metric("Gross yield", f"{gross_yield:.2f}%")
            st.metric("Net yield", f"{net_yield:.2f}%")
            st.metric("Annual profit", f"₹{annual_profit:,.0f}")

        with decision_col2:
            loan_amount = st.number_input("Loan amount (₹)", min_value=100000, max_value=80000000, value=2000000, step=50000)
            annual_interest_rate = st.number_input("Annual interest rate (%)", min_value=1.0, max_value=30.0, value=8.5, step=0.1)
            loan_years = st.slider("Loan tenure (years)", min_value=5, max_value=30, value=20)

            emi, total_interest, total_payable = calculate_emi(loan_amount, annual_interest_rate, loan_years)
            st.subheader("Mortgage affordability")
            st.metric("Estimated EMI", f"₹{emi:,.0f}/month")
            st.metric("Total interest", f"₹{total_interest:,.0f}")
            st.metric("Total payable", f"₹{total_payable:,.0f}")

            if loan_amount > 0:
                monthly_budget = monthly_rent if monthly_rent > 0 else 0
                if monthly_budget > 0:
                    st.info(f"This property can cover roughly {emi / monthly_budget:.2f}x of the monthly rental income in EMI terms.")
                else:
                    st.info("Enter a realistic rent estimate to compare mortgage cost against income.")

        st.markdown("---")
        st.subheader("Watchlist")
        watchlist_df = get_watchlist_summary()
        if watchlist_df.empty:
            st.info("No saved properties in your watchlist yet. Add one from the valuation tab.")
        else:
            st.dataframe(watchlist_df, width="stretch")

        st.markdown("---")
        st.subheader("Property comparison")
        compare_candidates = [
            {"label": f"{item.get('locality')} | {item.get('property_type')} | ₹{float(item.get('estimated_price_inr',0)):,.0f}", "value": item}
            for item in st.session_state.get("watchlist", [])
        ]
        if len(compare_candidates) >= 2:
            first_choice = st.selectbox("Property A", [item["label"] for item in compare_candidates], index=0)
            second_choice = st.selectbox("Property B", [item["label"] for item in compare_candidates], index=1 if len(compare_candidates) > 1 else 0)
            first_item = next(item["value"] for item in compare_candidates if item["label"] == first_choice)
            second_item = next(item["value"] for item in compare_candidates if item["label"] == second_choice)

            compare_df = pd.DataFrame([
                {
                    "Property": first_choice,
                    "Locality": first_item.get("locality"),
                    "Type": first_item.get("property_type"),
                    "BHK": first_item.get("bhk"),
                    "Price": float(first_item.get("estimated_price_inr", 0) or 0),
                    "Price/sqft": round(float(first_item.get("estimated_price_inr", 0) or 0) / float(first_item.get("builtup_area_sqft", 0) or 1), 2),
                },
                {
                    "Property": second_choice,
                    "Locality": second_item.get("locality"),
                    "Type": second_item.get("property_type"),
                    "BHK": second_item.get("bhk"),
                    "Price": float(second_item.get("estimated_price_inr", 0) or 0),
                    "Price/sqft": round(float(second_item.get("estimated_price_inr", 0) or 0) / float(second_item.get("builtup_area_sqft", 0) or 1), 2),
                },
            ])
            st.dataframe(compare_df, width="stretch")
            st.caption("Best value is usually the lower price per sq ft if the amenities and locality are similar.")
        else:
            st.info("Add at least two properties to the watchlist to compare them side by side.")

        st.markdown("---")
        st.subheader("Buy vs rent recommendation")
        buy_price = st.number_input("Purchase price (₹)", min_value=200000, max_value=100000000, value=3000000, step=50000)
        down_payment = st.number_input("Down payment (₹)", min_value=0, max_value=50000000, value=600000, step=50000)
        rent_amount = st.number_input("Current rent (₹/month)", min_value=0, max_value=200000, value=25000, step=1000)
        annual_maintenance_buy = st.number_input("Annual maintenance (₹)", min_value=0, max_value=1000000, value=20000, step=1000)
        years_stay = st.slider("Expected stay (years)", min_value=1, max_value=15, value=5)
        rate = st.number_input("Expected annual appreciation (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)

        verdict, reason, monthly_total_buy_cost, expected_value = get_buy_vs_rent_recommendation(
            buy_price,
            down_payment,
            annual_interest_rate,
            loan_years,
            rent_amount,
            annual_maintenance_buy,
            years_stay,
            rate,
        )

        st.metric("Recommendation", verdict)
        st.write(reason)
        st.write(f"Estimated total monthly buy cost: ₹{monthly_total_buy_cost:,.0f}")
        st.write(f"Estimated value after {years_stay} years: ₹{expected_value:,.0f}")

        st.markdown("---")
        st.subheader("Smart decision summary")
        decision_summary = pd.DataFrame(
            {
                "Metric": ["Locality quality", "Net rental yield", "Estimated EMI", "Annual profit"],
                "Value": [
                    f"{locality_quality['score']}/100 ({locality_quality['label']})",
                    f"{net_yield:.2f}%",
                    f"₹{emi:,.0f}/month",
                    f"₹{annual_profit:,.0f}",
                ],
            }
        )
        st.dataframe(decision_summary, width="stretch")

    with tabs[4]:
        st.markdown("### Local market analytics")
        analytics_locality = st.selectbox("Select locality", LOCALITIES)

        locality_price_trend = get_locality_price_trend(analytics_locality)
        demand_supply = get_demand_supply_score(analytics_locality)
        trend_df = get_locality_property_trends(analytics_locality)

        st.subheader("Price trend overview")
        if locality_price_trend.empty:
            st.info("No price trend available for this locality.")
        else:
            st.line_chart(locality_price_trend.set_index("Month")["Estimated_Price"])
            st.dataframe(locality_price_trend, width="stretch")

        st.markdown("---")
        st.subheader("Demand vs supply snapshot")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Demand score", f"{demand_supply['score']}/100")
        col_b.metric("Market label", demand_supply["label"])
        col_c.metric("Listings", demand_supply["listings"])
        st.info(demand_supply["reason"])

        st.markdown("---")
        st.subheader("Property mix by type")
        if trend_df.empty:
            st.info("No market data available for this locality.")
        else:
            st.bar_chart(trend_df.set_index("Property_Type")["avg_price"])
            st.dataframe(trend_df, width="stretch")

        st.markdown("### Smart local recommendations")
        local_recommendations = get_property_recommendations(analytics_locality)
        if local_recommendations.empty:
            st.info("No recommendations available.")
        else:
            st.dataframe(local_recommendations, width="stretch")

    with tabs[5]:
        history = fetch_user_history(st.session_state["username"])
        if history:
            report_df = pd.DataFrame(history)
            csv_data = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full History Report",
                data=csv_data,
                file_name=f"property_history_{st.session_state['username']}.csv",
                mime="text/csv",
            )
            st.dataframe(report_df, width="stretch")
        else:
            st.info("No records available for export.")

    st.sidebar.header("Project Summary")
    st.sidebar.markdown(
        """
        <div class="card">
            <p><strong>Dataset:</strong> Real property listings with local market fields</p>
            <p><strong>Model:</strong> Random Forest Regressor</p>
            <p><strong>Goal:</strong> Predict property prices and support real-estate valuation decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.write("This version includes decision-support tools like ROI, EMI, and locality scoring for a more realistic real-estate intelligence platform.")
