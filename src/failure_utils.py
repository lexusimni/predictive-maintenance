import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import json
import os

def plot_confusion(y_true, y_pred, labels=None, save_path=None, title="Confusion Matrix"):
    """
    Plots and optionally saves a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_metrics_to_json(metrics_dict, filename="evaluation_metrics.json", output_dir="../outputs/"):
    """
    Saves evaluation metrics to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved metrics to {path}")


def compare_models(models, X_train, X_test, y_train, y_test):
    """
    Trains multiple models and returns a comparison DataFrame (accuracy & AUC).
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {"accuracy": acc, "auc": auc}

    return pd.DataFrame(results).T.sort_values("auc", ascending=False)
