# train_model.py
# Run this ONCE to train and save your ML model as model.pkl
# After this, the FastAPI app will load model.pkl automatically.

import pickle
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save():
    print("Loading dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    print("Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # Save model
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    train_and_save()
