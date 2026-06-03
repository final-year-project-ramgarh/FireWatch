import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def load_data_and_model():
    """Load the dataset and model."""
    print("Loading data and model...")
    
    # Load the model and scaler
    model = joblib.load("best_fire_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Load the data
    years = [2022, 2023, 2024]
    dfs = []
    for year in years:
        try:
            df = pd.read_csv(f"modis_{year}_India.csv")
            df['year'] = year
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: modis_{year}_India.csv not found")
    
    if not dfs:
        raise FileNotFoundError("No data files found")
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Clean the data
    required_columns = ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']
    df = df.dropna(subset=required_columns)
    
    return df, model, scaler

def prepare_features(df, scaler):
    """Prepare features for prediction."""
    # Prepare features in the correct order expected by the model
    X = df[['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']].values
    X_scaled = scaler.transform(X)
    
    # Add special case for offshore fires
    offshore_mask = (df['latitude'] <= 10.0) | (df['longitude'] >= 92.0)
    
    return X_scaled, offshore_mask

def evaluate_model():
    """Evaluate the model's performance."""
    try:
        # Load data and model
        df, model, scaler = load_data_and_model()
        
        # Prepare features
        X, offshore_mask = prepare_features(df, scaler)
        
        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X)
        
        # Apply offshore fire correction
        y_pred[offshore_mask & (y_pred == 0)] = 3
        
        # Since we don't have true labels, we'll analyze the prediction distribution
        fire_types = {
            0: "🌳 Vegetation Fire",
            1: "🔥 Industrial/Urban Fire",
            2: "🏭 Other Static Land Source",
            3: "🌊 Offshore Fire"
        }
        
        # Calculate prediction distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        total = len(y_pred)
        
        print("\n🔥 Fire Type Distribution:")
        print("-" * 40)
        for fire_type, count in zip(unique, counts):
            percentage = (count / total) * 100
            print(f"{fire_types[fire_type]}: {count:,} samples ({percentage:.1f}%)")
        
        print("\n📊 Total samples:", f"{total:,}")
        
        # Since we don't have true labels, we can't calculate accuracy
        print("\n⚠️  Note: Ground truth labels are not available in the dataset.")
        print("To calculate accuracy, we would need labeled data with known fire types.")
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        labels = [fire_types[t] for t in unique]
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Fire Type Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('fire_type_distribution.png')
        print("\n✅ Saved fire type distribution plot as 'fire_type_distribution.png'")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_model()
