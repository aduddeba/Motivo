import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Encode categorical features
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    df[feature + '_encoded'] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# Normalize numeric features
scaler = StandardScaler()
df[numeric_features + [f + '_encoded' for f in categorical_features]] = scaler.fit_transform(df[numeric_features + [f + '_encoded' for f in categorical_features]])

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
    Returns list of (car_dict, probability) tuples, sorted by probability descending.
    """
    scores = []
    for _, car in df.iterrows():
        score = 0
        for pref, value in user_preferences.items():
            if pref in weights:
                if pref in numeric_features:
                    # For numeric, use difference
                    car_val = car[pref]
                    if pref == 'price':
                        # Normalize budget impact
                        budget = float(value) if value else car_val
                        score += weights[pref] * (budget - car_val) / budget if budget > 0 else 0
                    else:
                        score += weights[pref] * (car_val - float(value)) if value else 0
                elif pref + '_encoded' in car.index:
                    # For categorical, match
                    if car[pref + '_encoded'] == label_encoders[pref].transform([value])[0] if value in label_encoders[pref].classes_ else -1:
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

    # Convert to probabilities using softmax
    scores = np.array(scores)
    exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
    probabilities = exp_scores / np.sum(exp_scores)

    # Return top cars with probabilities
    car_probs = list(zip(df.to_dict('records'), probabilities))
    car_probs.sort(key=lambda x: x[1], reverse=True)
    return car_probs

def get_next_question(state):
    """
    Semi-adaptive question selection.
    Prioritize unanswered high-impact questions.
    """
    answered_questions = set([entry['q_id'] for entry in state.get('q_history', [])])
    all_questions = list(NODES.keys())
    high_impact = ['q_budget', 'q_body_commute', 'q_fuel_commute', 'q_transmission', 'q_doors']  # Example high-impact questions

    # Find unanswered high-impact questions
    for q in high_impact:
        if q not in answered_questions and q in all_questions:
            return q

    # Otherwise, pick a random unanswered question
    unanswered = [q for q in all_questions if q not in answered_questions]
    if unanswered:
        return random.choice(unanswered)

    return None  # All answered

# Load questions for get_next_question
with open("questions.json", "r") as f:
    QUESTION_TREE = json.load(f)
NODES = QUESTION_TREE["nodes"]