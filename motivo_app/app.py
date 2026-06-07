from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import joblib
import pandas as pd
import json
import random
import re

import comparison_engine as ce

app = Flask(__name__)
app.secret_key = "dreamcar_secret_key"

DATA_PATH = "data/cars.csv"
MODEL_PATH = "ml_model/car_data.pkl"

df, features, label_encoders, target_encoder = joblib.load(MODEL_PATH)
reverse_target_encoder = {v: k for k, v in target_encoder.items()}

df = pd.read_csv(DATA_PATH)
df.loc[94, 'price'] = 60000.0

with open("questions.json", "r") as f:
    QUESTION_TREE = json.load(f)

NODES = QUESTION_TREE["nodes"]
START = QUESTION_TREE["start"]


def recommend_car(answers):
    filtered = df.copy()
    for feature, value in answers.items():
        if feature not in df.columns or value is None:
            continue
        if feature == 'price':
            filtered = filtered[filtered[feature] <= float(value)]
        elif feature == 'mileage':
            val = float(value)
            if val > 0:
                filtered = filtered[filtered[feature] >= val]
        elif feature == 'drivetrain':
            filtered = filtered[filtered[feature].str.contains(value, case=False, na=False)]
        else:
            filtered = filtered[filtered[feature] == value]
    return filtered


def score_car(row, answers):
    """Score a car row against user answers — higher = closer match."""
    score = 0
    for feature, value in answers.items():
        if feature not in df.columns or value is None:
            continue
        if feature == 'price':
            car_price = row[feature]
            if pd.isna(car_price):
                continue
            budget = float(value)
            if car_price <= budget:
                score += 2
            elif car_price <= budget * 1.15:
                score += 1
        elif feature == 'mileage':
            val = float(value)
            if val <= 0:
                continue
            car_mpg = row[feature]
            if pd.isna(car_mpg):
                continue
            if car_mpg >= val:
                score += 2
            elif car_mpg >= val * 0.85:
                score += 1
        elif feature == 'drivetrain':
            if value.lower() in str(row[feature]).lower():
                score += 2
        else:
            if row[feature] == value:
                score += 2
    return score


# ===============================
# Routes
# ===============================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/quiz")
def quiz():
    session["answers"] = {}
    session["q_history"] = []
    session["current_q"] = START
    return render_template("quiz.html")


@app.route("/next_question")
def next_question():
    q_id = session.get("current_q", START)
    if q_id is None:
        return jsonify({"done": True})
    node = NODES[q_id].copy()
    node["id"] = q_id
    return jsonify(node)


@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    data = request.json
    q_id = data["question_id"]
    answer = data["answer"]

    node = NODES[q_id]

    value_map = node.get("value_map", {})
    mapped_value = value_map.get(answer, answer) if value_map else answer

    key = node.get("key")
    if key and mapped_value is not None:
        session["answers"][key] = mapped_value

    history = session.setdefault("q_history", [])
    history.append({"q_id": q_id, "key": key, "mapped_value": mapped_value})
    session.modified = True

    next_node = node.get("next")
    if isinstance(next_node, dict):
        next_q = next_node.get(answer)
    else:
        next_q = next_node

    # Skip mileage question for electric vehicles
    if next_q == "q_mileage" and session["answers"].get("fuel") == "Electric":
        next_q = None

    session["current_q"] = next_q
    session.modified = True

    if next_q is None:
        return jsonify({"done": True})

    next_data = NODES[next_q].copy()
    next_data["id"] = next_q
    return jsonify({"done": False, "question": next_data})


@app.route("/undo", methods=["POST"])
def undo():
    history = session.get("q_history", [])
    if not history:
        q_id = START
    else:
        last = history.pop()
        if last.get("key") and last["key"] in session.get("answers", {}):
            del session["answers"][last["key"]]
        q_id = last["q_id"]

    session["current_q"] = q_id
    session["q_history"] = history
    session.modified = True

    node = NODES[q_id].copy()
    node["id"] = q_id
    return jsonify({"question": node})


@app.route("/predict", methods=["POST"])
def predict():
    user_data = session.get("answers", {})
    matches = recommend_car(user_data)
    matches_clean = matches.fillna('')

    is_exact = not matches_clean.empty

    if is_exact:
        v_num = random.randint(0, len(matches_clean) - 1)
        selected = matches_clean.iloc[v_num]
    else:
        scores = df.apply(lambda row: score_car(row, user_data), axis=1)
        selected = df.loc[scores.idxmax()].fillna('')

    dream_car = re.sub(r"\b\d{4}\b", "", str(selected['name'])).strip()
    make = str(selected['make']).lower()
    model = str(selected['model']).lower().replace(" ", "-").replace("+", "")
    url = f"https://www.cars.com/research/{make}-{model}-2025/"

    return jsonify({
        "dream_car": dream_car,
        "url": url,
        "is_exact": is_exact
    })


@app.route("/result")
def result():
    car = request.args.get("car")
    url = request.args.get("url")
    if not car:
        return redirect(url_for("quiz"))
    return render_template("result.html", car=car, url=url)


@app.route("/car/<path:car_name>/compare")
def compare(car_name):
    selected = ce.find_car(car_name, df)
    if selected is None:
        return redirect(url_for("quiz"))

    competitors = ce.find_competitors(car_name, df, n=3)
    selected_name = ce.vehicle_display_name(selected)
    vehicle_names = [selected_name] + [ce.vehicle_display_name(row) for _, row in competitors.iterrows()]

    comparison_table = ce.build_comparison_table(selected, competitors)
    advantages, tradeoffs = ce.build_advantages_and_tradeoffs(selected, competitors)

    return render_template(
        "compare.html",
        selected_name=selected_name,
        vehicle_names=vehicle_names,
        comparison_table=comparison_table,
        summary=ce.generate_summary(selected, competitors),
        advantages=advantages,
        tradeoffs=tradeoffs,
        reasons=ce.explain_recommendation(selected, session.get("answers", {})),
        has_competitors=not competitors.empty,
    )


if __name__ == "__main__":
    app.run(debug=True)
