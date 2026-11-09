import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score

# ==========================================
# STEP 1: Load existing experiment results
# ==========================================

results_df = pd.read_pickle("results/predictions.pkl")
triage_df = pd.read_pickle("results/triage.pkl")

print("Predictions:", len(results_df))
print("Triage decisions:", len(triage_df))

print("\nPrediction results:")
print(results_df.head())

print("\nTriage results:")
print(triage_df.head())

# ==========================================
# STEP 2: Evaluate LLM triage performance
# ==========================================

print("\n========== LLM TRIAGE PERFORMANCE ==========")

positive_labels = {
    "diabetes": 1,
    "breast_cancer": 0, # The breast cancer dataset uses 0 = malignant, 1 = benign
    "heart": 1
}

for disease, positive_label in positive_labels.items():

    y_true = []
    y_pred = []

    # all successfully triaged patient
    for _, row in triage_df.iterrows():

        actual_label = row["labels"][disease]
        actual_disease = int(actual_label == positive_label)
        predicted_disease = int(disease in row["routes"])

        y_true.append(actual_disease)
        y_pred.append(predicted_disease)

    # the evaluation metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n=== {disease} ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")


    # make the confusion matrix.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No disease", "Disease present"])
    disp.plot(include_values=False)

    plt.setp(disp.ax_.get_yticklabels(), rotation=90, ha="center", va="center")


    ax = disp.ax_
    cell_labels = [
        [f"{cm[0,0]}\nTrue Negative", f"{cm[0,1]}\nFalse Positive"],
        [f"{cm[1,0]}\nFalse Negative", f"{cm[1,1]}\nTrue Positive"]
    ]

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cell_labels[i][j], ha="center", va="center", fontsize=9, fontweight="bold")

    plt.title(f"LLM Triage Confusion Matrix: {disease}")
    plt.tight_layout()
    plt.savefig(f"results/llm_triage_confusion_{disease}.png")
    plt.show()
    plt.close()
