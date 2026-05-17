import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dataset_builder import load_dataset, generate_synthetic_dataset
from ml_model import MLModelTrainer

def main():
    print("Loading dataset...")
    try:
        X, y = load_dataset()
    except Exception as e:
        print("Error loading dataset:", e)
        print("Generating synthetic dataset...")
        generate_synthetic_dataset(600, overwrite=True)
        X, y = load_dataset()

    print("Training models...")
    trainer = MLModelTrainer()
    trainer.train(X, y)
    print("Done. Check for confusion_matrix*.png files.")

if __name__ == "__main__":
    main()
