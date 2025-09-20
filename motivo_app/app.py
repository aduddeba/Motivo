from flask import Flask, render_template, request
import pandas as pd
import re

app = Flask(__name__)

# Load dataset once
df = pd.read_csv("vehicles_dataset.csv")


# Clean model names
def clean_model_name(model: str) -> str:
    forbidden_words = {
        "se", "le", "xe", "xle", "ls", "lxs", "ex", "limited", "ultimate",
        "advanced", "premium", "standard", "range", "touring", "sport",
        "luxury", "pure", "electric", "twin", "performance", "plus", "base",
        "select", "light", "long", "gt", "a-spec", "edition", "platinum",
        "sel", "2lt", "lt", "sxt", "sv", "s", "14t", "l", "latitude",
        "preferred", "active", "sle", "r/t"
    }

    words = model.lower().split()
    filtered_words = [
        w for w in words
        if w not in forbidden_words and not re.match(r"\d+(\.\d+)", w)
    ]
    return "-".join(filtered_words)


# Home page with form
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        min_price = int(request.form["min_price"])
        max_price = int(request.form["max_price"])
        fuel = request.form["fuel"]
        body = request.form["body"]
        min_rating = float(request.form["min_rating"])

        filtered_df = df[
            (df["price"] >= min_price) &
            (df["price"] <= max_price) &
            (df["fuel"].str.lower() == fuel.lower()) &
            (df["body"].str.lower() == body.lower())
            # (df["rating"] >= min_rating)   # Uncomment if rating column exists
        ]

        # Deduplicate by make + model
        visited, results = set(), []
        for item in filtered_df["name"]:
            parts = item.strip().split()
            if len(parts) < 2:
                continue

            car = (parts[0], parts[1], " ".join(parts[2:]))
            if (parts[0], parts[1]) not in visited:
                make, model = car[1], car[2]
                model_url = clean_model_name(model)
                url = f"https://cars.usnews.com/cars-trucks/{make.lower()}/{model_url}"
                results.append({"make": make, "model": model, "url": url})
                visited.add((parts[0], parts[1]))

        return render_template("results.html", results=results)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
