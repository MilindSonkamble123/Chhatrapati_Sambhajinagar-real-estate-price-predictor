from pathlib import Path
import random

import pandas as pd

random.seed(42)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_FILE = DATA_DIR / "real_estate_chhatrapati_sambhajinagar.csv"

locality_factors = {
    "CIDCO": 1.18,
    "Shivaji Nagar": 1.12,
    "N-6": 1.10,
    "Harsul": 1.04,
    "Panchakki": 1.08,
    "Madhuban Colony": 1.05,
    "Jalna Road": 1.00,
    "Waluj": 0.88,
    "MIDC": 0.95,
    "Kanchanwadi": 1.03,
    "Garkhadi": 0.92,
    "Cidco N-6": 1.17,
    "Samarth Nagar": 1.09,
    "Aundh": 1.15,
    "Rahatani": 0.90,
}

property_type_factors = {
    "Apartment": 1.02,
    "Flat": 1.00,
    "Villa": 1.35,
    "Bungalow": 1.45,
    "Plot": 0.78,
}

localities = list(locality_factors.keys())
property_types = list(property_type_factors.keys())


def generate_row():
    locality = random.choice(localities)
    property_type = random.choice(property_types)

    area_sq_ft = random.randint(420, 2600)
    bedrooms = random.choices([1, 2, 3, 4, 5], weights=[0.12, 0.30, 0.32, 0.18, 0.08])[0]
    bathrooms = random.choices([1, 2, 3, 4], weights=[0.18, 0.42, 0.28, 0.12])[0]
    age_of_property_years = random.randint(0, 28)
    distance_from_city_center_km = round(random.uniform(1.5, 18.5), 2)
    floor_no = random.randint(1, 16)
    total_floors = max(floor_no, random.randint(2, 20))
    parking = random.choices([0, 1, 2], weights=[0.12, 0.68, 0.20])[0]
    furnished = random.choices([True, False], weights=[0.42, 0.58])[0]
    security = random.choices([True, False], weights=[0.75, 0.25])[0]
    amenities_count = random.randint(0, 8)

    locality_score = locality_factors[locality]
    property_score = property_type_factors[property_type]

    base_price = (
        area_sq_ft * 0.045
        + bedrooms * 18
        + bathrooms * 14
        + amenities_count * 7
        + parking * 19
        + (15 if furnished else 0)
        + (18 if security else 0)
        + (total_floors * 1.5)
        - age_of_property_years * 2.2
        - distance_from_city_center_km * 1.8
    )

    price_in_lakhs = base_price * locality_score * property_score
    price_in_lakhs += random.uniform(-10, 20)
    price_in_lakhs = max(12.5, round(price_in_lakhs, 2))

    row = {
        "locality": locality,
        "property_type": property_type,
        "area_sq_ft": area_sq_ft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_of_property_years": age_of_property_years,
        "distance_from_city_center_km": distance_from_city_center_km,
        "floor_no": floor_no,
        "total_floors": total_floors,
        "parking": parking,
        "furnished": furnished,
        "security": security,
        "amenities_count": amenities_count,
        "price_in_lakhs": price_in_lakhs,
    }
    return row


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    records = [generate_row() for _ in range(1800)]
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset created: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Price range: ₹{df['price_in_lakhs'].min():.2f} L to ₹{df['price_in_lakhs'].max():.2f} L")


if __name__ == "__main__":
    main()
