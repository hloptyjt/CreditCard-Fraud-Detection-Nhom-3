from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import time

def get_models():
    """Trả về danh sách mô hình"""
    pos_weight = 492 / (284315 - 492)  # Tính scale_pos_weight
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42),
        "CatBoost": CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_state=42)
    }
    return models

def train_with_smote(model, X_train, y_train):
    """Huấn luyện với SMOTE"""
    pipeline = ImbPipeline([('smote', SMOTE(random_state=42)), ('model', model)])
    pipeline.fit(X_train, y_train)
    return pipeline