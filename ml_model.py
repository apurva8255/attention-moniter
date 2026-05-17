"""
ml_model.py
Trains Naive Bayes, Random Forest, SVM, and Logistic Regression.
Compares metrics, saves the best model with joblib.
"""

import os
import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

MODEL_PATH = "best_model.joblib"
SCALER_PATH = "scaler.joblib"
RESULTS_PATH = "model_comparison.csv"


class MLModelTrainer:
    """Trains and evaluates multiple ML classifiers."""

    LABEL_MAP = {"High": 2, "Medium": 1, "Low": 0}
    LABEL_INV = {2: "High", 1: "Medium", 0: "Low"}

    def __init__(self):
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.comparison_results = []
        self._is_trained = False

    def _get_classifiers(self):
        return {
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
            "SVM": SVC(kernel="rbf", C=1.5, probability=True, random_state=42),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, solver="lbfgs"),
        }

    def train(self, X: list, y: list, callback=None) -> dict:
        """
        Train all classifiers. Returns comparison dict.

        Args:
            X: feature matrix (list of lists)
            y: labels (list of strings: High/Medium/Low)
            callback: optional function(model_name, metrics) called per model
        """
        if len(X) < 20:
            raise ValueError("Need at least 20 samples to train.")

        X_np = np.array(X, dtype=float)
        y_enc = np.array([self.LABEL_MAP.get(lbl, 1) for lbl in y])

        X_scaled = self.scaler.fit_transform(X_np)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

        classifiers = self._get_classifiers()
        results = {}
        best_f1 = -1.0
        
        self._pending_cms = {}
        self._pending_cms_norm = {}

        for name, clf in classifiers.items():
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
                cm_norm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2], normalize='true')
                
                self._pending_cms[name] = cm
                self._pending_cms_norm[name] = cm_norm
                
                metrics = {
                    "model": name,
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                    "confusion_matrix": cm.tolist(),
                }
                results[name] = metrics

                if f1 > best_f1:
                    best_f1 = f1
                    self.best_model = clf
                    self.best_model_name = name

                if callback:
                    callback(name, metrics)

                print(f"[ML] {name}: acc={acc:.3f} prec={prec:.3f} "
                      f"rec={rec:.3f} f1={f1:.3f}")

            except Exception as e:
                print(f"[ML] Error training {name}: {e}")
                results[name] = {"model": name, "accuracy": 0, "precision": 0,
                                 "recall": 0, "f1_score": 0, "error": str(e)}

        self.comparison_results = list(results.values())
        self._save_comparison(self.comparison_results)
        self._save_best_model()
        self._save_confusion_matrices()
        self._is_trained = True

        print(f"[ML] Best model: {self.best_model_name} (F1={best_f1:.3f})")
        return results

    def _save_comparison(self, results: list):
        import csv
        with open(RESULTS_PATH, "w", newline="") as f:
            if not results:
                return
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    def _save_best_model(self):
        if self.best_model is not None:
            joblib.dump(self.best_model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            print(f"[ML] Saved best model -> {MODEL_PATH}")

    def _save_confusion_matrices(self):
        if not hasattr(self, '_pending_cms') or not self._pending_cms:
            return
        
        labels = ["Low", "Medium", "High"]
        name = self.best_model_name
        
        # 1. Standard CM for Best Model
        if name in self._pending_cms:
            disp = ConfusionMatrixDisplay(confusion_matrix=self._pending_cms[name], display_labels=labels)
            disp.plot(cmap="Blues", colorbar=False)
            plt.title(f"Confusion Matrix — {name}")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png", dpi=120)
            plt.close()
            
            # 2. Normalized CM for Best Model
            disp_norm = ConfusionMatrixDisplay(confusion_matrix=self._pending_cms_norm[name], display_labels=labels)
            disp_norm.plot(cmap="Blues", colorbar=False, values_format=".2f")
            plt.title(f"Normalized Confusion Matrix — {name}")
            plt.tight_layout()
            plt.savefig("confusion_matrix_normalized.png", dpi=120)
            plt.close()

        # 3. Per-Model Confusion Matrix
        num_models = len(self._pending_cms)
        fig, axes = plt.subplots(1, num_models, figsize=(4 * num_models, 4))
        if num_models == 1:
            axes = [axes]
            
        for ax, (model_name, cm) in zip(axes, self._pending_cms.items()):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(model_name)
            
        plt.tight_layout()
        plt.savefig("confusion_matrix_all_models.png", dpi=120)
        plt.close()

    def load_model(self) -> bool:
        """Load pre-trained model from disk."""
        if os.path.isfile(MODEL_PATH) and os.path.isfile(SCALER_PATH):
            try:
                self.best_model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self._is_trained = True
                print("[ML] Loaded saved model.")
                return True
            except Exception as e:
                print(f"[ML] Load error: {e}")
        return False

    def predict(self, feature_vector: list) -> tuple:
        """
        Predict attention label.

        Returns:
            (label_str, confidence_0_to_1)
        """
        if not self._is_trained or self.best_model is None:
            return "Medium", 0.5

        try:
            X = np.array([feature_vector], dtype=float)
            X_scaled = self.scaler.transform(X)
            pred = self.best_model.predict(X_scaled)[0]
            label = self.LABEL_INV.get(int(pred), "Medium")

            confidence = 0.5
            if hasattr(self.best_model, "predict_proba"):
                proba = self.best_model.predict_proba(X_scaled)[0]
                confidence = float(max(proba))

            return label, confidence
        except Exception as e:
            print(f"[ML] Predict error: {e}")
            return "Medium", 0.5

    def is_trained(self) -> bool:
        return self._is_trained

    def get_comparison_results(self) -> list:
        """Return list of model metric dicts."""
        if self.comparison_results:
            return self.comparison_results
        # Try loading from CSV
        if os.path.isfile(RESULTS_PATH):
            import csv
            results = []
            with open(RESULTS_PATH) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
            return results
        return []
