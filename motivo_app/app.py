from flask import Flask, render_template, request
import pandas as pd
import re

app = Flask(__name__)

# Load dataset once
df = pd.read_csv("vehicles_dataset.csv")


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

        filtered_df = df[
            (df["price"] >= min_price) &
            (df["price"] <= max_price) &
            (df["fuel"].str.lower() == fuel.lower()) &
            (df["body"].str.lower() == body.lower())
        ]
        if make != "Any":
            filtered_df = filtered_df[(df["make"].str.lower() == make.lower())]

        # Rank cars before building results
        ranked_df = rank_cars(filtered_df)

        # Deduplicate by make + model
        visited, results = set(), []
        for _, row in ranked_df.iterrows():
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
                    "score": row["score"]
                })
                visited.add((parts[0], parts[1]))

        return render_template("results.html", results=results)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
