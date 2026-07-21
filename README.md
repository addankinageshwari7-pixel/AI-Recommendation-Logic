# AI Recommendation System

A professional, responsive AI-based recommendation web application built with Python, Flask, Scikit-learn, and Bootstrap 5.

## Project Goal
This intelligent recommendation system suggests items based on user preferences. It calculates the Cosine Similarity between user-provided interests and the dataset of items (using TF-IDF Vectorization) to return the top recommendations along with confidence scores.

## Features
- **Professional UI**: Light theme with blue accents, fully responsive using Bootstrap 5. No animations as per requirements.
- **AI Recommendation Engine**: Built using Scikit-learn to match interests with items.
- **Analytics Dashboard**: Simple analytics with Chart.js.
- **Admin Panel**: Manage the dataset and retrain the ML model dynamically.
- **Preferences Form**: Clean form to gather user keywords and categories.

## Prerequisites
- Python 3.8+ installed on your system.

## Setup Instructions

1. **Install Dependencies**
   Navigate to the project directory and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Initial Model**
   Run the model trainer to generate `recommendation_model.pkl` from `dataset/items.csv`:
   ```bash
   python model_trainer.py
   ```

3. **Run the Application**
   Start the Flask server:
   ```bash
   python app.py
   ```

4. **Access the App**
   Open your browser and navigate to:
   http://127.0.0.1:5000/

## Project Structure
- `app.py`: Main Flask application.
- `model_trainer.py`: Script to generate the TF-IDF vectorizer and item matrix.
- `recommendation_model.pkl`: The serialized Machine Learning model.
- `dataset/items.csv`: The sample dataset of items.
- `templates/`: HTML templates (Jinja2).
- `static/`: CSS and JS assets.

## Troubleshooting
- If you see "Dataset missing", upload a new CSV file via the Admin Panel and click "Retrain Model".
