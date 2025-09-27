from flask import Flask, render_template, request
import pandas as pd
import re
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt

app = Flask(__name__)

# ===============================
# Load dataset and clean data
# ===============================
df = pd.read_csv("vehicles_dataset.csv")
df.dropna(inplace=True)
df.columns = df.columns.str.strip().str.lower()

# ===============================
# Clustering setup
# ===============================
features = ["price", "mileage", "cylinders", "year", "doors"]
X = df[features].copy()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster (k chosen from elbow method)
kmeans = KMeans(n_clusters=5, random_state=42)
df["category"] = kmeans.fit_predict(X_scaled)

# Mapping of cluster numbers -> labels
cluster_labels = {
    0: "Luxury Efficient",
    1: "Performance Sports",
    2: "Luxury Premium",
    3: "Budget Commuter",
    4: "Used Premium"
}

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
            (df["fuel"].str.lower() == fuel.lower()) &
            (df["body"].str.lower() == body.lower())
        ]
        if make != "Any":
            filtered_df = filtered_df[df["make"].str.lower() == make.lower()]

        scored_df = rank_cars(filtered_df)

        # Deduplicate by make + model
        visited, results = set(), []
        for _, row in scored_df.iterrows():
            parts = row["name"].strip().split()
            if len(parts) < 2:
                continue

            car = (parts[0], parts[1], " ".join(parts[2:]))
            if (parts[0], parts[1]) not in visited:
                make, model = car[1], car[2]
                model_url = clean_model_name(model)
                url = f"https://cars.usnews.com/cars-trucks/{make.lower()}/{model_url}"
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
