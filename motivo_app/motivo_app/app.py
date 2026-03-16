from flask import Flask, render_template, request, jsonify, session
import joblib
import pandas as pd
import json
import random
from flask import redirect, url_for
import re

app = Flask(__name__)
app.secret_key = "dreamcar_secret_key"  # needed for session storage

# ===============================
# Load dataset and encoders
# ===============================
DATA_PATH = "data/cars.csv"
MODEL_PATH = "ml_model/car_data.pkl"  # saved dataset + encoders

# Load dataset + encoders
df, features, label_encoders, target_encoder = joblib.load(MODEL_PATH)
reverse_target_encoder = {v: k for k, v in target_encoder.items()}

df = pd.read_csv(DATA_PATH)

df.loc[94, 'price'] = 60000.0 # Fixing a missing price value

# Load questions
with open("questions.json", "r") as f:
    QUESTIONS = json.load(f)

def recommend_car(answers, df=df, features=features):
    filtered = df.copy()
    for feature, value in answers.items():
        if feature not in df.columns:
            continue
        if feature in ['price']:
            value = float(value)
            filtered = filtered[filtered[feature] <= value]
            
        elif feature in ['mileage']:
            filtered = filtered[filtered[feature] >= float(value)]
        elif feature in ['fuel']:
            '''
            if (value == 'Gasoline'):
                value = 0
            elif (value == 'Hybrid'):
                value = 1
            else:
                value = 2
            '''
            filtered = filtered[filtered[feature] == value]
        elif feature in ['body']:
            '''
            if (value == 'SUV'):
                value = 0
            elif (value == 'Pickup Truck'):
                value = 1
            elif (value == 'Sedan'):
                value = 2
            elif (value == 'Passenger Van'):
                value = 3
            elif (value == 'Cargo Van'):
                value = 4
            elif (value == 'Hatchback'):
                value = 5
            elif (value == 'Convertible'):
                value = 6
            else:
                value = 7
            '''
            filtered = filtered[filtered[feature] == value]
        '''
        elif feature in label_encoders:
            value = label_encoders[feature].get(value)

            if value is None:
                continue
            filtered = filtered[filtered[feature] == value]
        '''
        

    return filtered



# ===============================
# Routes
# ===============================

@app.route("/")
def index():
    """Landing page"""
    return render_template("index.html")

@app.route("/quiz")
def quiz():
    """Start the quiz"""
    session["answers"] = {}
    return render_template("quiz.html")

@app.route("/get_question/<int:q_id>")
def get_question(q_id):
    """Send one question at a time"""
    if q_id < len(QUESTIONS):
        return jsonify(QUESTIONS[q_id])
    else:
        return jsonify({"end": True})

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    """Save user answer in session"""
    data = request.json
    q_key = data["key"]
    answer = data["answer"]
    session["answers"][q_key] = answer
    session.modified = True

    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    user_data = session.get("answers", {})
    matches = recommend_car(user_data)
    matches_clean = matches.fillna('')

    if matches_clean.empty:
        dream_car = None
        message = "No exact match found. Showing some options."
        matches_list = []
        url = None
    else:
        # Randomly select one car from the matches
        v_num = random.randint(0, len(matches_clean) - 1)
        dream_car = "TEST_CAR"
        #re.sub(r"\b\d{4}\b", "", matches_clean.iloc[v_num]['name']).strip()
        message = None
        matches_list = matches_clean.to_dict(orient="records")

        # --- Extract make and model safely ---

        make, model = matches_clean.iloc[v_num]['make'], matches_clean.iloc[v_num]['model']
        model = "".join(model).lower()
        model = model.replace(" ", "-").replace("+", "") if model else ""

        # --- Construct cars.com URL ---
        url = f"https://www.cars.com/research/{make.lower()}-{model}-2025/"

    return jsonify({
            "dream_car": dream_car,
            "matches": matches_list,
            "message": message,
            "url": url
    })




@app.route("/result")
def result():
    car = request.args.get("car")
    url = request.args.get("url")

    # If somehow car is missing, redirect back to quiz
    if not car:
        return "No car specified. Go back and retake the quiz.", 400

    return render_template("result.html", car=car, url=url)

if __name__ == "__main__":
    app.run(debug=True)
