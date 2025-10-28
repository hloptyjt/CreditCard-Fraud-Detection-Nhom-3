from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def find_best_threshold(y_true, y_proba):
    """Tìm ngưỡng tối ưu theo F1"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Không gian lận', 'Gian lận'],
                yticklabels=['Không gian lận', 'Gian lận'])
    plt.title(title)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.tight_layout()
    plt.show()
    return cm

def plot_roc_pr(y_true, y_proba, threshold):
    plt.figure(figsize=(12,5))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_proba):.4f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.subplot(1,2,2)
    plt.plot(recall, precision)
    idx = np.argmin(np.abs(_ - threshold))
    plt.scatter(recall[idx], precision[idx], c='red', s=100, label=f'Thresh = {threshold:.2f}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.tight_layout()
    plt.show()