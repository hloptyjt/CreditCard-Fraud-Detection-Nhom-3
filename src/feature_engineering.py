from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def apply_pca(X_train, n_components=2):
    """Giảm chiều bằng PCA"""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_train)
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")
    return X_pca, pca

def plot_pca(X_pca, y_train):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.6, s=1)
    plt.title(f"PCA 2D Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label='Class')
    plt.show()

def apply_tsne(X_train, sample_size=5000):
    """Giảm chiều bằng t-SNE (mẫu nhỏ)"""
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_train[idx])
    return X_tsne, idx