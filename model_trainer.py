import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

def train_model():
    dataset_path = os.path.join('dataset', 'items.csv')
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return False
        
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Combine features for content-based filtering
    df['combined_features'] = df['category'] + " " + df['description']
    
    print("Training TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
    print("Saving model and data...")
    # Save the vectorizer, the matrix, and the dataframe
    model_data = {
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'dataframe': df
    }
    
    joblib.dump(model_data, 'recommendation_model.pkl')
    print("Model saved successfully as recommendation_model.pkl")
    return True

if __name__ == "__main__":
    train_model()
