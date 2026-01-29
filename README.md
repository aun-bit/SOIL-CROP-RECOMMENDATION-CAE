# Soil–Crop Recommendation System using Convolutional Autoencoder (CAE)

This project predicts the most suitable crop for a given soil–climate profile using a CAE-based classifier and a Logistic Regression baseline on the Kaggle Crop Recommendation dataset.
## How to Run (Local)

1. Create a Python venv and install:
     ```bash
   pip install -r requirements.txt
   
2. Train CAE model:
     ```bash
    python train.py
    
3. Evaluate:
    ```bash
   python evaluate.py
   
4. Baseline and inference:
    ```bash
   python baseline_logreg.py
   python inference.py --input_json sample_input.json --top_k 3


