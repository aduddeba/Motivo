from flask import Flask, render_template, request, jsonify, session
import joblib
import pandas as pd
import json
import random

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

# Load questions
with open("questions.json", "r") as f:
    QUESTIONS = json.load(f)

def recommend_car(answers, df=df, features=features):
    filtered = df.copy()
    print("User answers:", answers)
    for feature, value in answers.items():
        if feature not in df.columns:
            continue
        if feature in ['price']:
            value = float(value)
            filtered = filtered[filtered[feature] <= value]
            print("\n\n\n\n\nPrice:", filtered)
            
        elif feature in ['doors']:
            filtered = filtered[filtered[feature] == float(value)]
        elif feature in ['fuel']:
            if (value == 'Gasoline'):
                value = 0
            elif (value == 'Hybrid'):
                value = 1
            else:
                value = 2
            filtered = filtered[filtered[feature] == value]
        elif feature in ['body']:
            print("\n\n\nEncoding:", value, filtered['body'])
            if (value == 'SUV'):
                value = 0
            elif (value == 'Pickup Truck'):
                value = 1
            elif (value == 'Sedan'):
                value = 2
            filtered = filtered[filtered[feature] == value]
        '''
        elif feature in label_encoders:
            value = label_encoders[feature].get(value)

            if value is None:
                continue
            filtered = filtered[filtered[feature] == value]
        '''
        

    # Optional debug print with decoded names
    if 'name' in filtered.columns:
        filtered['name'] = filtered['name'].map(reverse_target_encoder)

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
    try:
        user_data = session.get("answers", {})
        matches = recommend_car(user_data)
        matches_clean = matches.fillna('')
        print("Matches found:", len(matches))

        # --- Decode numeric car names back to strings ---
        #if target_column in matches.columns:
            #matches[target_column] = matches[target_column].map(reverse_target_encoder)

        if matches_clean.empty:
            dream_car = None
            message = "No exact match found. Showing some options."
            matches_list = []
        else:
            # Random car selected from matches
            dream_car = matches_clean.iloc[random.randint(0, len(matches) - 1)]['name']
            print("Dream car:", dream_car)
            message = None
            matches_list = matches_clean.to_dict(orient="records")

        return jsonify({
            "dream_car": dream_car,
            "matches": matches_list,
            "message": message
        })

    except Exception as e:
        print("Error in /predict:", e)
        return jsonify({"error": str(e), "matches": [], "dream_car": None}), 500




@app.route("/result")
def result():
    """Render result page"""
    return render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True)
