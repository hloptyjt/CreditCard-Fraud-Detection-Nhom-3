import pandas as pd
import numpy as np
import time
from src.preprocessing import *
from src.eda import *
from src.feature_engineering import *
from src.model_training import *
from src.evaluation import *

def main():
    print("BẮT ĐẦU DỰ ÁN PHÁT HIỆN GIAN LẬN THẺ TÍN DỤNG")
    
    # 1. Đọc dữ liệu
    df = load_data("data/creditcard.csv")
    
    # 2. EDA
    print("\nTỷ lệ gian lận:", df['Class'].mean() * 100, "%")
    plot_class_distribution(df)
    plot_correlation_matrix(df)
    
    # 3. Tiền xử lý
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X[['Time', 'Amount']], _ = scale_time_amount(X.copy())
    X_scaled, _ = scale_full(X)
    
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # 4. Huấn luyện mô hình
    models = get_models()
    results = {}
    
    print("\nHUẤN LUYỆN MÔ HÌNH...")
    for name, model in models.items():
        start = time.time()
        pipeline = train_with_smote(model, X_train, y_train)
        
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)
        
        thresh, best_f1 = find_best_threshold(y_test, y_proba)
        y_pred_opt = (y_proba >= thresh).astype(int)
        
        auc = roc_auc_score(y_test, y_proba)
        plot_confusion_matrix(y_test, y_pred_opt, f"{name} (thresh={thresh:.2f})")
        
        results[name] = {'AUC': auc, 'F1': best_f1, 'Time': time.time() - start}
        print(f"{name}: AUC={auc:.4f}, F1={best_f1:.4f}, Time={results[name]['Time']:.1f}s")
    
    # 5. Kết quả tốt nhất
    best_model = max(results, key=lambda x: results[x]['AUC'])
    print(f"\nMÔ HÌNH TỐT NHẤT: {best_model}")
    
    # 6. PCA + t-SNE
    X_pca, _ = apply_pca(X_train)
    plot_pca(X_pca, y_train)
    
    # 7. Feature importance (Random Forest)
    rf_pipeline = train_with_smote(models['Random Forest'], X_train, y_train)
    importances = pd.DataFrame({
        'Feature': df.drop('Class', axis=1).columns,
        'Importance': rf_pipeline.named_steps['model'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 6 đặc trưng:")
    print(importances.head(6))
    plot_feature_distributions(df, importances.head(4)['Feature'].tolist())

if __name__ == "__main__":
    main()