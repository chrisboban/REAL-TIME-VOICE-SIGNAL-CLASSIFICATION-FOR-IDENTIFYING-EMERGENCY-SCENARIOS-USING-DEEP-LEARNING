import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = load_model('D:/SEM 8/codes/model1.keras')

data_val = np.load("val_data.npz")
X_val, y_val = data_val["X"], data_val["y"]

predictions = model.predict(X_val)
predicted_labels = (predictions > 0.7).astype(int)

test_loss, test_acc = model.evaluate(X_val, y_val, verbose=1)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

precision = precision_score(y_val, predicted_labels)
recall = recall_score(y_val, predicted_labels)
f1 = f1_score(y_val, predicted_labels)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, predicted_labels, target_names=["Non-Emergency", "Emergency"]))

cm = confusion_matrix(y_val, predicted_labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Emergency", "Emergency"], yticklabels=["Non-Emergency", "Emergency"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
