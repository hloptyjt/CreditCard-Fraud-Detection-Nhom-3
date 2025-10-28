import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_class_distribution(df):
    """Vẽ phân bố lớp"""
    plt.figure(figsize=(8, 6))
    colors = ["#0101DF", "#DF0101"]
    ax = sns.countplot(x='Class', data=df, palette=colors)
    plt.title("Phân bố Lớp (0: Không Gian Lận | 1: Gian Lận)")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom')
    plt.show()

def plot_correlation_matrix(df):
    """Vẽ ma trận tương quan"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
    plt.title("Ma trận tương quan các đặc trưng")
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(df, top_features):
    """Vẽ KDE cho các đặc trưng quan trọng"""
    fraud = df[df['Class'] == 1]
    nonfraud = df[df['Class'] == 0]
    
    plt.figure(figsize=(14, 10))
    for i, feat in enumerate(top_features, 1):
        plt.subplot(2, 2, i)
        sns.kdeplot(nonfraud[feat], label='Không gian lận', fill=True, alpha=0.6)
        sns.kdeplot(fraud[feat], label='Gian lận', fill=True, alpha=0.6, color='red')
        plt.title(f"Phân bố {feat}")
        plt.legend()
    plt.tight_layout()
    plt.show()