🚗 Motivo
# Find your dream car — powered by machine learning.
Motivo is a web application that recommends vehicles based on your personal preferences. Answer a few simple questions about your budget, fuel type, body style, and fuel efficiency expectations, and Motivo uses a clustering-based ML model to match you with the cars that fit your lifestyle.

How It Works

Tell us what you want — Motivo walks you through a short quiz covering budget, fuel preference (electric, hybrid, or gasoline), body type, and minimum MPG.
ML does the matching — Your answers are scaled and fed into a pre-trained clustering model (vehicle_cluster.pkl) built with scikit-learn. The model groups thousands of vehicles into clusters and finds the ones closest to your ideal profile.
Get your results — Motivo returns a curated list of vehicles that match your criteria, displayed in a clean web interface.

Tech Stack
LayerTechnologyBackendPython, FlaskML / Datascikit-learn, pandas, NumPy, joblibFrontendHTML, CSS, JavaScript (Jinja2 templates)DeploymentGunicorn
Project Structure
Motivo/
├── app.py                  # Flask application entry point
├── questions.json          # Quiz questions served to the frontend
├── requirements.txt        # Python dependencies
├── scaler.pkl              # Pre-fitted StandardScaler for input normalization
├── vehicle_cluster.pkl     # Pre-trained vehicle clustering model
├── model_dev.ipynb         # Jupyter notebook for model development & EDA
├── data/                   # Vehicle datasets used for training
├── ml_model/               # Model training scripts and utilities
├── motivo_app/             # Application package / modules
├── templates/              # Jinja2 HTML templates
└── static/                 # CSS, JavaScript, and image assets
Getting Started
Prerequisites

Python 3.9+

Installation
bash# Clone the repository
git clone https://github.com/aduddeba/Motivo.git
cd Motivo

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
Running Locally
bash# Development server
python app.py

# — or with Gunicorn (production) —
gunicorn app:app
Then open http://localhost:5000 in your browser.
Model Development
The model_dev.ipynb notebook contains the full pipeline for data exploration, feature engineering, and model training. If you want to retrain or tweak the clustering model, start there.
Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

License
This project does not currently specify a license. Contact the repository owner for usage terms.
