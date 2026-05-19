
import pandas as pd
from river import ensemble, metrics
from pathlib import Path

CSV_PATH = Path(r"c:\Users\90530\Desktop\barış\dataset\incart_streaming.csv")

def verify_model_logic():
    if not CSV_PATH.exists():
        print("CSV not found!")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples.")
    print("Class counts:", df['label'].value_counts().to_dict())

    model = ensemble.AdaptiveRandomForestClassifier(n_models=3, seed=42)
    acc = metrics.Accuracy()
    
    feature_cols = [c for c in df.columns if c != 'label']
    
    # Test on first 1000 samples
    for i in range(1000):
        row = df.iloc[i]
        x = row[feature_cols].to_dict()
        y = row['label']
        
        y_pred = model.predict_one(x)
        if y_pred is not None:
            acc.update(y, y_pred)
        
        model.learn_one(x, y)
        
    print(f"Accuracy after 1000 samples: {acc.get():.2%}")
    if acc.get() > 0.7:
        print("Model learning logic verified.")
    else:
        print("Accuracy lower than expected, check feature mapping.")

if __name__ == "__main__":
    verify_model_logic()
