import pandas as pd
import joblib
import os

DATA_PATH = os.path.join("data", "cars.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "car_data.pkl")

print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

features = ['price', 'fuel', 'body']
target = 'name'

# ------------------------------
# Ensure numeric columns are numbers
# ------------------------------
numeric_features = ['price']  # add other numeric features if needed
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing required values
df = df.dropna(subset=features + [target])
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# ------------------------------
# Encode categorical features
# ------------------------------
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != target:
        le = {val: i for i, val in enumerate(df[col].unique())}
        df[col] = df[col].map(le)
        label_encoders[col] = le

# Encode target separately
target_encoder = {val: i for i, val in enumerate(df[target].unique())}
df[target] = df[target].map(target_encoder)

# ------------------------------
# Save cleaned and encoded dataset
# ------------------------------
joblib.dump((df, features, label_encoders, target_encoder), MODEL_PATH)
print(f"✅ Dataset saved to {MODEL_PATH}")

# ------------------------------
# Simple filtering function (no fuzzy matching)
# ------------------------------
def recommend_car(answers, df=df, features=features):
    """
    answers: dictionary like {'price': 30000, 'fuel': 'Gasoline', 'body': 'SUV'}
    Returns the filtered cars matching user criteria exactly.
    """
    filtered = df.copy()

    for feature, value in answers.items():
        # Convert numeric input to proper type
        if feature in numeric_features:
            value = float(value)
            filtered = filtered[filtered[feature] <= value]  # e.g., price less than or equal
        # Encode categorical answers
        if feature in label_encoders:
            value = label_encoders[feature].get(value)
            filtered = filtered[filtered[feature] == value]
        
    return filtered

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    sample_answers = {'price': 40000, 'fuel': 'Gasoline', 'body': 'Sedan'}
    matches = recommend_car(sample_answers)
    print("Filtered matches:")
    print(matches)
