from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Global variables for model
vectorizer = None
tfidf_matrix = None
dataset = None

def load_model():
    global vectorizer, tfidf_matrix, dataset
    model_path = 'recommendation_model.pkl'
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        vectorizer = model_data['vectorizer']
        tfidf_matrix = model_data['tfidf_matrix']
        dataset = model_data['dataframe']
        return True
    return False

# Initialize model
load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    if request.method == 'POST':
        category = request.form.get('category', '')
        interests = request.form.get('interests', '')
        
        # Combine user input for vectorization
        user_input = category + " " + interests
        
        return redirect(url_for('recommendation', query=user_input))
        
    categories = []
    if dataset is not None:
        categories = dataset['category'].unique().tolist()
    return render_template('preferences.html', categories=categories)

@app.route('/recommendation')
def recommendation():
    query = request.args.get('query', '')
    if not query or dataset is None:
        flash("Please enter your preferences or ensure model is trained.")
        return redirect(url_for('preferences'))
        
    # Transform query
    user_vec = vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    # Get top 5 indices
    top_indices = similarities.argsort()[-5:][::-1]
    
    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score > 0:
            item = dataset.iloc[idx].to_dict()
            item['score'] = round(score * 100, 2)
            results.append(item)
            
    return render_template('recommendation.html', results=results, query=query)

@app.route('/dashboard')
def dashboard():
    total_users = 150 # Mock data
    total_recommendations = 420 # Mock data
    
    category_counts = {}
    if dataset is not None:
        category_counts = dataset['category'].value_counts().to_dict()
        
    most_selected_category = ""
    if category_counts:
        most_selected_category = max(category_counts, key=category_counts.get)
        
    return render_template('dashboard.html', 
                         total_users=total_users, 
                         total_recommendations=total_recommendations,
                         category_counts=category_counts,
                         most_selected_category=most_selected_category)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'retrain':
            # Run model trainer
            import model_trainer
            success = model_trainer.train_model()
            if success:
                load_model()
                flash("Model retrained successfully!", "success")
            else:
                flash("Failed to retrain model. Dataset missing.", "danger")
                
        elif action == 'upload':
            if 'dataset_file' not in request.files:
                flash("No file uploaded", "danger")
                return redirect(request.url)
            file = request.files['dataset_file']
            if file.filename == '':
                flash("No selected file", "danger")
                return redirect(request.url)
            if file and file.filename.endswith('.csv'):
                filename = secure_filename('items.csv')
                filepath = os.path.join('dataset', filename)
                file.save(filepath)
                flash("Dataset uploaded successfully. Please retrain model.", "success")
                
        elif action == 'delete':
            filepath = os.path.join('dataset', 'items.csv')
            if os.path.exists(filepath):
                os.remove(filepath)
                flash("Dataset deleted.", "warning")
                
    dataset_records = []
    if dataset is not None:
        dataset_records = dataset.to_dict('records')
        
    return render_template('admin.html', records=dataset_records)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
