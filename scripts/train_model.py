
import os
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from cnn_model import build_cnn

# Load dataset
print("Loading dataset...")
full_df = pd.read_pickle("Dataset/full_dataset.pkl")

# Prepare input features — stack 3 signals as channels
print("Preparing features...")
X, y, participants = [], [], []
for _, row in full_df.iterrows():
    flow   = row["flow"]
    thorac = row["thorac"]
    spo2   = resample(row["spo2"], 960)
    window = np.stack([flow, thorac, spo2], axis=1)
    X.append(window)
    y.append(row["label"])
    participants.append(row["participant"])

X            = np.array(X)
participants = np.array(participants)

# Encode labels
le        = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Leave-One-Participant-Out Cross Validation
participants_list = sorted(full_df["participant"].unique())
all_results = []

for test_pid in participants_list:
    print(f"\nFold: Test on {test_pid}")

    test_mask  = participants == test_pid
    train_mask = ~test_mask

    X_train, y_train = X[train_mask], y_encoded[train_mask]
    X_test,  y_test  = X[test_mask],  y_encoded[test_mask]

    print(f"  Train: {X_train.shape[0]} windows | Test: {X_test.shape[0]} windows")

    class_weights     = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    model = build_cnn()
    model.fit(X_train, y_train, epochs=20, batch_size=64,
              class_weight=class_weight_dict, verbose=0)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    all_results.append({"participant": test_pid, "accuracy": acc,
                        "precision": prec, "recall": rec, "cm": cm})

# Final summary
print("\n" + "="*55)
print(f"{"Participant":<15} {"Accuracy":<12} {"Precision":<12} {"Recall":<12}")
print("="*55)
accs, precs, recs = [], [], []
for r in all_results:
    print(f"{r["participant"]:<15} {r["accuracy"]:<12.4f} {r["precision"]:<12.4f} {r["recall"]:<12.4f}")
    accs.append(r["accuracy"])
    precs.append(r["precision"])
    recs.append(r["recall"])
print("="*55)
print(f"{"Average":<15} {np.mean(accs):<12.4f} {np.mean(precs):<12.4f} {np.mean(recs):<12.4f}")

print("\nConfusion Matrices (0=Hypopnea, 1=Normal, 2=Obstructive Apnea):")
for r in all_results:
    print(f"\n{r["participant"]}:")
    print(r["cm"])
