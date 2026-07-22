import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import json
import os
import random

# Load data
df = pd.read_csv("data/cars.csv")

# Define features
numeric_features = ['price', 'mileage', 'cylinders']
categorical_features = ['fuel', 'body', 'drivetrain']

# Handle missing values
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
df[categorical_features] = df[categorical_features].fillna('Unknown')

# Some rows have data-entry errors in mileage (MPG) far outside any real
# vehicle's rating — e.g. 9711, 5581, 4723 — likely odometer readings
# mistakenly entered in this column. Cap at 150, comfortably above the
# highest legitimate MPGe rating, so a single bad row can't dominate scoring.
df['mileage'] = df['mileage'].clip(upper=150)

# Encode categorical features
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    df[feature + '_encoded'] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# Note: numeric_features (price, mileage, cylinders) are intentionally left
# unscaled — the weighted-scoring formulas below compare them directly
# against raw user-entered thresholds (dollar budget, MPG minimum).

# For simplicity, use weighted scoring instead of training a model
# Weights can be adjusted based on importance
weights = {
    'price': -0.3,  # Lower price better
    'mileage': 0.4,  # Higher MPG better
    'cylinders': -0.2,  # Fewer cylinders better (efficiency)
    'fuel_encoded': 0.1,  # Preference for certain fuels
    'body_encoded': 0.2,
    'drivetrain_encoded': 0.1
}

def get_car_scores(user_preferences: dict) -> list:
    """
    Calculate scores for all cars based on user preferences.
    Returns list of (car_dict, match_fraction) tuples, sorted by match_fraction descending.
    """
    scores = []
    for _, car in df.iterrows():
        score = 0
        for pref, value in user_preferences.items():
            if pref in numeric_features and pref in weights:
                # For numeric, use difference
                car_val = car[pref]
                if pref == 'price':
                    # Normalize budget impact as % over/under budget so it's
                    # comparable in magnitude to the other weighted terms.
                    # weights['price'] is negative ("lower price better"), so
                    # this must be (car_val - budget) to penalize going over
                    # budget and reward coming in under it — matching the
                    # (car_val - threshold) convention used for every other
                    # numeric feature below.
                    budget = float(value) if value else car_val
                    if budget > 0:
                        score += weights[pref] * (car_val - budget) / budget
                else:
                    # 0 (or blank) means "no preference" for these fields
                    if value and float(value) != 0:
                        score += weights[pref] * (car_val - float(value))
            elif pref + '_encoded' in weights and pref + '_encoded' in car.index:
                # For categorical, match
                if value in label_encoders[pref].classes_ and car[pref + '_encoded'] == label_encoders[pref].transform([value])[0]:
                    score += weights[pref + '_encoded']
            elif pref == 'transmission':
                if value and value.lower() in str(car['transmission']).lower():
                    score += 0.1
            elif pref == 'doors':
                try:
                    if car['doors'] == int(value):
                        score += 0.1
                except (ValueError, TypeError):
                    pass
        scores.append(score)

    # Scale raw scores to a 0-1 "match" fraction. Softmax over ~1000 cars
    # would treat this as picking one winner out of the whole dataset, which
    # crushes every score down near 1/len(df) regardless of fit quality. A
    # min-max scale instead reflects how close each car is to the best
    # possible match for these specific preferences.
    scores = np.array(scores)
    min_score, max_score = scores.min(), scores.max()
    if max_score > min_score:
        match_fractions = (scores - min_score) / (max_score - min_score)
    else:
        match_fractions = np.ones_like(scores)

    # Return top cars with match fractions
    car_probs = list(zip(df.to_dict('records'), match_fractions))
    car_probs.sort(key=lambda x: x[1], reverse=True)
    return car_probs

def get_next_question(state):
    """
    Semi-adaptive question selection.
    Prioritize unanswered high-impact questions.
    """
    answered_questions = set([entry['q_id'] for entry in state.get('q_history', [])])
    answered_keys = set([entry['key'] for entry in state.get('q_history', []) if entry.get('key')])
    all_questions = list(NODES.keys())
    high_impact = ['q_budget', 'q_body_commute', 'q_fuel_commute', 'q_transmission', 'q_doors']  # Example high-impact questions

    def already_covered(q):
        # Skip questions already asked, or whose key was already answered
        # by a different question (avoids asking a re-worded duplicate).
        if q in answered_questions:
            return True
        key = NODES[q].get('key')
        return key is not None and key in answered_keys

    # Find unanswered high-impact questions
    for q in high_impact:
        if q in all_questions and not already_covered(q):
            return q

    # Otherwise, pick a random unanswered question
    unanswered = [q for q in all_questions if not already_covered(q)]
    if unanswered:
        return random.choice(unanswered)

    return None  # All answered

# Load questions for get_next_question
with open("questions.json", "r") as f:
    QUESTION_TREE = json.load(f)
NODES = QUESTION_TREE["nodes"]