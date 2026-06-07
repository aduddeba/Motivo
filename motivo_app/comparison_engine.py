"""
Comparison engine for Motivo's "Compare Competitors" feature.

Given a recommended vehicle, this module finds the closest competing vehicles
already present in cars.csv (using only columns that exist in the dataset)
and prepares everything needed to render a side-by-side comparison: a
feature-matched competitor list, a comparison table with "best value"
highlights, a short natural-language summary, and advantage/tradeoff call-outs.
"""

import re

import numpy as np
import pandas as pd

# Columns used to build similarity vectors. All exist in cars.csv.
# Horsepower / cargo space / seating capacity are NOT present in the dataset
# and are intentionally left out rather than invented.
NUMERIC_FEATURES = ["price", "mileage", "cylinders", "doors"]
CATEGORICAL_FEATURES = ["body", "fuel", "drivetrain"]

# Fields shown in the comparison table, in display order.
# "best" marks which direction ("min"/"max") counts as the standout value,
# or None when there's no universally "better" direction (e.g. body style).
COMPARISON_FIELDS = [
    {"key": "price", "label": "Price", "kind": "currency", "best": "min"},
    {"key": "mileage", "label": "MPG", "kind": "number", "best": "max"},
    {"key": "engine", "label": "Engine", "kind": "text", "best": None},
    {"key": "cylinders", "label": "Cylinders", "kind": "number", "best": None},
    {"key": "fuel", "label": "Fuel Type", "kind": "text", "best": None},
    {"key": "transmission", "label": "Transmission", "kind": "text", "best": None},
    {"key": "body", "label": "Body Style", "kind": "text", "best": None},
    {"key": "doors", "label": "Doors", "kind": "number", "best": None},
    {"key": "drivetrain", "label": "Drivetrain", "kind": "text", "best": None},
]


def strip_year(name):
    """'2024 Toyota RAV4 Prime XSE' -> 'Toyota RAV4 Prime XSE' (mirrors app.py's dream_car formatting)."""
    return re.sub(r"\b\d{4}\b", "", str(name)).strip()


def _to_float(value):
    try:
        if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def find_car(car_name, cars_df):
    """Look up the row in cars_df whose year-stripped name matches car_name (case-insensitive)."""
    cars_df = cars_df.reset_index(drop=True)
    stripped = cars_df["name"].apply(strip_year).str.lower()
    target = strip_year(car_name).strip().lower()
    if not target:
        return None

    matches = cars_df.index[stripped == target]
    if len(matches) == 0:
        matches = cars_df.index[stripped.str.contains(re.escape(target), na=False)]
    if len(matches) == 0:
        return None
    return cars_df.loc[matches[0]]


def _feature_matrix(cars_df):
    """Build a similarity matrix: z-scored numeric specs + one-hot categorical specs."""
    numeric = cars_df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median())
    spread = numeric.std(ddof=0).replace(0, 1)
    numeric_norm = (numeric - numeric.mean()) / spread

    categorical = pd.get_dummies(cars_df[CATEGORICAL_FEATURES].fillna("Unknown"))

    matrix = pd.concat(
        [numeric_norm.reset_index(drop=True), categorical.reset_index(drop=True)],
        axis=1,
    )
    return matrix.fillna(0).to_numpy(dtype=float)


def find_competitors(car_name, cars_df, n=3):
    """
    Return the n vehicles most similar to car_name.

    Each vehicle is converted into a feature vector built from its price,
    MPG, cylinder count, door count, body style, fuel type and drivetrain,
    and vehicles are ranked by cosine similarity to the target. Other
    trims/years of the same make+model are skipped so results read as
    genuine competitors rather than variants of the same car.
    """
    cars_df = cars_df.reset_index(drop=True)
    target_row = find_car(car_name, cars_df)
    if target_row is None:
        return cars_df.iloc[0:0]

    target_idx = target_row.name
    target_key = (str(target_row["make"]).lower(), str(target_row["model"]).lower())

    matrix = _feature_matrix(cars_df)
    target_vector = matrix[target_idx]
    target_norm = np.linalg.norm(target_vector)
    if target_norm == 0:
        return cars_df.iloc[0:0]

    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms[row_norms == 0] = 1e-9
    similarity = (matrix @ target_vector) / (row_norms * target_norm)

    ranked = pd.Series(similarity, index=cars_df.index).sort_values(ascending=False)

    competitor_idx = []
    seen_models = {target_key}
    for idx in ranked.index:
        if idx == target_idx:
            continue
        key = (str(cars_df.loc[idx, "make"]).lower(), str(cars_df.loc[idx, "model"]).lower())
        if key in seen_models:
            continue
        seen_models.add(key)
        competitor_idx.append(idx)
        if len(competitor_idx) == n:
            break

    return cars_df.loc[competitor_idx]


def vehicle_display_name(row):
    return strip_year(row.get("name", ""))


def _format_value(value, kind):
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if kind in ("currency", "number"):
        number = _to_float(value)
        if number is None:
            return str(value)
        if kind == "currency":
            return f"${number:,.0f}"
        return f"{number:.0f}" if number == int(number) else f"{number:g}"
    return str(value)


def build_comparison_table(selected_row, competitor_df):
    """
    Build [{label, values, best_index}, ...] for the comparison table, where
    values[0] is always the selected vehicle. A field is omitted entirely if
    every vehicle in the comparison is missing it.
    """
    vehicles = [selected_row] + [row for _, row in competitor_df.iterrows()]
    rows = []

    for field in COMPARISON_FIELDS:
        key = field["key"]
        if key not in selected_row.index:
            continue

        raw_values = [vehicle.get(key) for vehicle in vehicles]
        if all(v is None or v == "" or (isinstance(v, float) and pd.isna(v)) for v in raw_values):
            continue

        best_index = None
        if field["best"] in ("min", "max"):
            numeric_values = [(i, _to_float(v)) for i, v in enumerate(raw_values)]
            valid = [(i, v) for i, v in numeric_values if v is not None]
            if valid:
                picker = min if field["best"] == "min" else max
                best_index = picker(valid, key=lambda pair: pair[1])[0]

        rows.append({
            "label": field["label"],
            "values": [_format_value(v, field["kind"]) for v in raw_values],
            "best_index": best_index,
        })

    return rows


def generate_summary(selected_row, competitor_df):
    """Compose a short, data-derived natural-language comparison summary."""
    selected_name = vehicle_display_name(selected_row)
    competitor_names = [vehicle_display_name(row) for _, row in competitor_df.iterrows()]

    if not competitor_names:
        return f"No close competitors were found for the {selected_name} in the current lineup."

    comp_phrase = (
        competitor_names[0] if len(competitor_names) == 1
        else ", ".join(competitor_names[:-1]) + f" and {competitor_names[-1]}"
    )

    clauses = []

    selected_mpg = _to_float(selected_row.get("mileage"))
    competitor_mpgs = competitor_df["mileage"].apply(_to_float).dropna()
    if selected_mpg is not None and not competitor_mpgs.empty:
        if selected_mpg >= competitor_mpgs.max():
            clauses.append("offers the best fuel economy of the group")
        elif selected_mpg <= competitor_mpgs.min():
            leader = vehicle_display_name(competitor_df.loc[competitor_mpgs.idxmax()])
            clauses.append(f"trails the {leader} on fuel economy")

    selected_price = _to_float(selected_row.get("price"))
    competitor_prices = competitor_df["price"].apply(_to_float).dropna()
    if selected_price is not None and not competitor_prices.empty:
        if selected_price <= competitor_prices.min():
            clauses.append("comes in at the most competitive price")
        elif selected_price >= competitor_prices.max():
            cheapest = vehicle_display_name(competitor_df.loc[competitor_prices.idxmin()])
            clauses.append(f"carries a higher price tag than the {cheapest}")
        else:
            clauses.append("is priced right in the middle of the pack")

    if clauses:
        summary = f"Compared with the {comp_phrase}, the {selected_name} " + " and ".join(clauses[:2]) + "."
    else:
        summary = (
            f"Compared with the {comp_phrase}, the {selected_name} holds its own with a "
            "similar overall profile based on the available specs."
        )

    selected_drivetrain = str(selected_row.get("drivetrain") or "").strip()
    other_drivetrains = sorted({
        str(d).strip() for d in competitor_df["drivetrain"].dropna()
        if str(d).strip() and str(d).strip() != selected_drivetrain
    })
    if selected_drivetrain and other_drivetrains:
        summary += (
            f" Note that some competitors come in {', '.join(other_drivetrains)} "
            f"rather than {selected_drivetrain}."
        )

    return summary


def build_advantages_and_tradeoffs(selected_row, competitor_df):
    """Derive short, data-backed 'Major Advantages' / 'Major Tradeoffs' bullet lists."""
    advantages = []
    tradeoffs = []

    selected_price = _to_float(selected_row.get("price"))
    competitor_prices = competitor_df["price"].apply(_to_float).dropna()
    if selected_price is not None and not competitor_prices.empty:
        if selected_price < competitor_prices.min():
            advantages.append("Lower price than every listed competitor")
        elif selected_price > competitor_prices.max():
            tradeoffs.append("Priced higher than every listed competitor")

    selected_mpg = _to_float(selected_row.get("mileage"))
    competitor_mpgs = competitor_df["mileage"].apply(_to_float).dropna()
    if selected_mpg is not None and not competitor_mpgs.empty:
        if selected_mpg > competitor_mpgs.max():
            advantages.append("Best fuel economy (MPG) in the comparison")
        elif selected_mpg < competitor_mpgs.min():
            tradeoffs.append("Lower fuel economy (MPG) than its competitors")

    selected_cyl = _to_float(selected_row.get("cylinders"))
    competitor_cyls = competitor_df["cylinders"].apply(_to_float).dropna()
    if selected_cyl is not None and not competitor_cyls.empty:
        if selected_cyl < competitor_cyls.min():
            advantages.append("Smaller, more efficient engine (fewer cylinders)")
        elif selected_cyl > competitor_cyls.max():
            advantages.append("More powerful engine option (more cylinders)")

    selected_drivetrain = str(selected_row.get("drivetrain") or "")
    competitor_drivetrains = competitor_df["drivetrain"].dropna().astype(str)
    if ("All-wheel" in selected_drivetrain or "Four-wheel" in selected_drivetrain) and not competitor_drivetrains.empty:
        if not competitor_drivetrains.str.contains("All-wheel|Four-wheel", case=False, regex=True).all():
            advantages.append("Standard all/four-wheel drive where some competitors are 2WD only")

    if not advantages:
        advantages.append("Delivers a well-balanced spec sheet relative to its competitors")
    if not tradeoffs:
        tradeoffs.append("No standout weaknesses found among the compared specs")

    return advantages, tradeoffs


# Maps a quiz answer key (see questions.json) to a human-readable reason and a
# check against the recommended vehicle's row.
_RECOMMENDATION_RULES = [
    {
        "key": "price",
        "label": "Fits your budget",
        "check": lambda value, row: (lambda budget, price: budget is not None and price is not None and price <= budget)(
            _to_float(value), _to_float(row.get("price"))
        ),
    },
    {
        "key": "body",
        "label": "Matches your preferred body style",
        "check": lambda value, row: str(row.get("body") or "").strip().lower() == str(value).strip().lower(),
    },
    {
        "key": "fuel",
        "label": "Matches your fuel type preference",
        "check": lambda value, row: str(row.get("fuel") or "").strip().lower() == str(value).strip().lower(),
    },
    {
        "key": "drivetrain",
        "label": "Aligns with your drivetrain preference",
        "check": lambda value, row: str(value).strip().lower() in str(row.get("drivetrain") or "").lower(),
    },
    {
        "key": "mileage",
        "label": "Meets your fuel economy goals",
        "check": lambda value, row: (lambda target, mpg: target is not None and target > 0 and mpg is not None and mpg >= target)(
            _to_float(value), _to_float(row.get("mileage"))
        ),
    },
]


def explain_recommendation(selected_row, answers):
    """Translate the quiz answers (session['answers']) into plain-language reasons this car was suggested."""
    if not answers:
        return []

    reasons = []
    for rule in _RECOMMENDATION_RULES:
        value = answers.get(rule["key"])
        if value is None:
            continue
        try:
            if rule["check"](value, selected_row):
                reasons.append(rule["label"])
        except (TypeError, ValueError):
            continue
    return reasons
