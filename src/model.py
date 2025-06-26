
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve,
                             roc_auc_score)
import joblib

class StudentPlacementModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.best_threshold = 0.5
        self.results_df = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        X = df.drop('placement', axis=1)
        y = df['placement']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.load_data()

        # Scale the data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        youden_index = tpr - fpr
        self.best_threshold = thresholds[np.argmax(youden_index)]

        y_pred_optimal = (y_proba >= self.best_threshold).astype(int)

        self.results_df = X_test.copy()
        self.results_df['Actual Placement'] = y_test.values
        self.results_df['Predicted Probability'] = y_proba
        self.results_df['Predicted (0.5)'] = (y_proba >= 0.5).astype(int)
        self.results_df[f'Predicted ({self.best_threshold:.2f})'] = y_pred_optimal
        self.results_df['Correct (0.5)'] = self.results_df['Actual Placement'] == self.results_df['Predicted (0.5)']
        self.results_df[f'Correct ({self.best_threshold:.2f})'] = self.results_df['Actual Placement'] == y_pred_optimal

        return {
            "accuracy_0.5": accuracy_score(y_test, self.results_df['Predicted (0.5)']),
            "accuracy_best": accuracy_score(y_test, y_pred_optimal),
            "classification_report": classification_report(y_test, y_pred_optimal),
            "best_threshold": self.best_threshold
        }

    def predict(self, cgpa, iq, use_best_threshold=True):
        input_df = pd.DataFrame([[cgpa, iq]], columns=['cgpa', 'iq'])
        input_scaled = self.scaler.transform(input_df)
        prob = self.model.predict_proba(input_scaled)[0, 1]
        threshold = self.best_threshold if use_best_threshold else 0.5
        prediction = int(prob >= threshold)
        return {
            "cgpa": cgpa,
            "iq": iq,
            "predicted_probability": prob,
            "threshold_used": threshold,
            "placement_prediction": prediction
        }

    def save_model(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

