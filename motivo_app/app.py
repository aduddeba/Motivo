from flask import Flask, render_template, request
import pandas as pd
import re
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
from urllib.parse import quote_plus
#import matplotlib.pyplot as plt

app = Flask(__name__)

# ===============================
# Load dataset and clean data
# ===============================
df = pd.read_csv("vehicles_dataset.csv")
df.dropna(inplace=True)
df.columns = df.columns.str.strip().str.lower()

# Remove trim from the name for easy URL generation
def strip_trim_from_name(row, name_col="name", trim_col="trim"):
    name = str(row.get(name_col, "") or "")
    trim = row.get(trim_col, "")
    if pd.isna(trim) or trim is None:
        return name

    trim = str(trim).strip()
    if trim == "":
        return name

    # escape regex metacharacters in trim
    esc = re.escape(trim)

    # pattern:
    #  - match " - TRIM" or "— TRIM" or ": TRIM"
    #  - match "(TRIM)"
    #  - match trim as a whole word (so single-letter 'S' won't match 'Soul')
    pattern = rf'(?i)(?:\s*[-–—:]\s*{esc}|\s*\(\s*{esc}\s*\)|\b{esc}\b)'

    # remove occurrences that match the pattern
    new_name = re.sub(pattern, ' ', name)

    # collapse multiple spaces and trim leading/trailing whitespace
    new_name = re.sub(r'\s+', ' ', new_name).strip()

    return new_name

df["name_trim"] = df.apply(strip_trim_from_name, axis=1)

# ===============================
# Clustering setup
# ===============================
features = ["price", "mileage", "cylinders", "doors", "body"]
X = df[features].copy()

# One-hot encode 'body'
X = pd.get_dummies(X, columns=["body"], drop_first=True)

# Save encoded feature names
encoded_features = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster (k chosen from elbow method)
kmeans = KMeans(n_clusters=6, random_state=42)
df["category"] = kmeans.fit_predict(X_scaled)

# Add encoded columns back into df
for col in X.columns:
    df[col] = X[col]

# Mapping of cluster numbers -> labels
cluster_labels = {
    0: "Standard Premium SUV",
    1: "High-End Truck/Performance Vehicle",
    2: "New Sedan (Commuter/Mid-Range)",
    3: "New Luxury Convertible",
    4: "New Efficient Hatchback/Compact",
    5: "New Premium Van/High-Power Utility"
}

# Apply mapping
df["category_label"] = df["category"].map(cluster_labels)

# ===============================
# Helper functions
# ===============================

# Clean model names
def clean_model_name(model: str) -> str:
    forbidden_words = {
        "se", "le", "xe", "xle", "ls", "lxs", "ex", "xlt", "lx", "limited", "ultimate",
        "advanced", "premium", "standard", "range", "touring", "sport",
        "luxury", "pure", "electric", "twin", "performance", "plus", "base",
        "select", "light", "long", "gt", "a-spec", "edition", "platinum",
        "sel", "2lt", "lt", "sxt", "sv", "s", "14t", "l", "latitude",
        "preferred", "active", "sle", "r/t", "denali", "wilderness", "turbo", "altitude",
        "pursuit", "3lt", "luxe", "eawd", "awd", "laredo", "dynamic"
    }

    words = model.lower().split()
    filtered_words = [
        w for w in words
        if w not in forbidden_words and not re.match(r"\d+(\.\d+)", w)
    ]
    return "-".join(filtered_words)


# Simple heuristic ranker
def rank_cars(df):
    if df.empty:
        return df
    
    # Normalize columns safely
    if "price" in df.columns:
        df["price_norm"] = (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min() + 1e-9)
    else:
        df["price_norm"] = 0

    if "rating" in df.columns:
        df["rating_norm"] = (df["rating"] - df["rating"].min()) / (df["rating"].max() - df["rating"].min() + 1e-9)
    else:
        df["rating_norm"] = 0

    if "mileage" in df.columns:
        df["mileage_norm"] = (df["mileage"] - df["mileage"].min()) / (df["mileage"].max() - df["mileage"].min() + 1e-9)
    else:
        df["mileage_norm"] = 0

    # Weighted scoring: cheaper, newer, higher rated cars are ranked higher
    df["score"] = (
        100 * (
            (0.5 * (1 - df["price_norm"])) +
            (0.4 * df["rating_norm"]) +
            (0.1 * (1 - df["mileage_norm"]))  # use 1 - mileage_norm so lower mileage ranks higher
        )
    )

    df["score"] = df["score"].fillna(0).astype(int)
    
    return df.sort_values("score", ascending=False)



# Home page with form
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        min_price = int(request.form["min_price"])
        max_price = int(request.form["max_price"])
        fuel = request.form["fuel"]
        body = request.form["body"]
        make = request.form["make"]

        # Filter
        filtered_df = df[
            (df["price"] >= min_price) &
            (df["price"] <= max_price) &
            (df["fuel"].str.lower() == fuel.lower())
        ]
        if make != "Any":
            filtered_df = filtered_df[df["make"].str.lower() == make.lower()]
        if body != "Any":
            filtered_df = filtered_df[df["body"].str.lower() == body.lower()]

        scored_df = rank_cars(filtered_df)

        # Deduplicate by make + model
        visited, results = set(), []
        for _, row in scored_df.iterrows():
            parts = row["name_trim"].strip().split()
            if len(parts) < 2:
                continue

            car = (parts[0], parts[1], " ".join(parts[2:]))
            make, model = car[1], car[2]
            model_url = clean_model_name(model)
            url = f"https://www.cars.com/research/{make.lower()}-{model_url}-2025/"

            if (parts[0], parts[1]) not in visited:
                results.append({
                    "make": make,
                    "model": model,
                    "url": url,
                    "score": row["score"],
                    "category": row["category_label"]  # 🚀 show ML group instead of score
                })
                visited.add((parts[0], parts[1]))

        return render_template("results.html", results=results)

    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)
