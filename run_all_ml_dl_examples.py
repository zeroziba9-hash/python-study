import json
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

results = {}


def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -----------------------------------------------------------------------------
# 0) Intro examples
# -----------------------------------------------------------------------------
section("0) 서문 예제")
results["intro"] = {
    "factory_failure": "ML",
    "cctv_detection": "DL",
    "churn_prediction": "ML",
}
print(results["intro"])


# -----------------------------------------------------------------------------
# 1) 준비 파트
# -----------------------------------------------------------------------------
section("1) 준비 파트")
try:
    import torch

    device_info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
except Exception:
    device_info = {
        "torch_version": None,
        "cuda_available": False,
        "device": "CPU",
    }

results["prep"] = {
    "recommended_device": "RTX 3060 (if available)",
    "cifar100": {
        "classes": 100,
        "image_size": "32x32 RGB",
        "train": 50000,
        "test": 10000,
    },
    "runtime_device": device_info,
}
print(json.dumps(results["prep"], ensure_ascii=False, indent=2))


# -----------------------------------------------------------------------------
# 2) 개념 파트
# -----------------------------------------------------------------------------
section("2) 개념 파트")
results["concepts"] = {
    "supervised": "레이블 있는 데이터",
    "reinforcement": "보상 최대화",
    "deep_learning": "신경망 기반 표현학습",
    "containment": "AI ⊃ ML ⊃ DL",
}
print(results["concepts"])


# -----------------------------------------------------------------------------
# 3) 머신러닝 파트
# -----------------------------------------------------------------------------
section("3) 머신러닝 파트")

# 3-1 Logistic Regression
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_s, y_train)
pred = clf.predict(X_test_s)
ml_log = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
}
print("Logistic Regression accuracy:", ml_log["accuracy"])

# 3-2 Regression comparison
Xr, yr = load_diabetes(return_X_y=True)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)
lin = LinearRegression().fit(Xr_train, yr_train)
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(Xr_train, yr_train)
lin_pred = lin.predict(Xr_test)
rf_pred = rf.predict(Xr_test)
ml_reg = {
    "LinearRegression": {
        "MAE": float(mean_absolute_error(yr_test, lin_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(yr_test, lin_pred))),
    },
    "RandomForest": {
        "MAE": float(mean_absolute_error(yr_test, rf_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(yr_test, rf_pred))),
    },
}
print("Regression metrics:", ml_reg)

# 3-3 SVM GridSearch
Xw, yw = load_wine(return_X_y=True)
Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    Xw, yw, test_size=0.2, random_state=42, stratify=yw
)
pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
param_grid = {"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.01], "svm__kernel": ["rbf"]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(Xw_train, yw_train)
svm_pred = grid.best_estimator_.predict(Xw_test)
ml_svm = {
    "best_params": grid.best_params_,
    "best_cv": float(grid.best_score_),
    "test_acc": float(accuracy_score(yw_test, svm_pred)),
}
print("SVM GridSearch:", ml_svm)

# 3-4 Feature importance
bc = load_breast_cancer()
Xb = pd.DataFrame(bc.data, columns=bc.feature_names)
yb = bc.target
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.2, random_state=42, stratify=yb
)
rf_cls = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_cls.fit(Xb_train, yb_train)
imp = pd.Series(rf_cls.feature_importances_, index=Xb.columns).sort_values(ascending=False)
ml_imp = imp.head(5).to_dict()
print("Top-5 features:", ml_imp)

results["ml"] = {
    "logistic": ml_log,
    "regression": ml_reg,
    "svm_gridsearch": ml_svm,
    "feature_importance_top5": ml_imp,
}


# -----------------------------------------------------------------------------
# 4) 딥러닝 파트 (실행 가능 버전)
# -----------------------------------------------------------------------------
section("4) 딥러닝 파트")
Xd, yd = load_digits(return_X_y=True)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.2, random_state=42, stratify=yd
)

# MLP baseline
mlp = Pipeline([
    ("scaler", StandardScaler()),
    (
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=30,
            random_state=42,
        ),
    ),
])
mlp.fit(Xd_train, yd_train)
d_pred = mlp.predict(Xd_test)
mlp_acc = float(accuracy_score(yd_test, d_pred))
print("Digits MLP accuracy:", mlp_acc)

# Regularization sweep (alpha ~= weight decay)
reg_rows = []
for alpha in [0.0, 1e-4, 1e-3]:
    m = Pipeline([
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=alpha,
                max_iter=25,
                random_state=42,
            ),
        ),
    ])
    m.fit(Xd_train, yd_train)
    p = m.predict(Xd_test)
    reg_rows.append({"alpha": alpha, "acc": float(accuracy_score(yd_test, p))})

# Early-stopping comparison
m_no_es = Pipeline([
    ("scaler", StandardScaler()),
    (
        "mlp",
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=40, random_state=42),
    ),
])
m_no_es.fit(Xd_train, yd_train)
a_no_es = float(accuracy_score(yd_test, m_no_es.predict(Xd_test)))

m_es = Pipeline([
    ("scaler", StandardScaler()),
    (
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=40,
            early_stopping=True,
            n_iter_no_change=5,
            validation_fraction=0.1,
            random_state=42,
        ),
    ),
])
m_es.fit(Xd_train, yd_train)
a_es = float(accuracy_score(yd_test, m_es.predict(Xd_test)))

results["dl"] = {
    "mlp_digits_acc": mlp_acc,
    "regularization_sweep": reg_rows,
    "early_stopping": {"without_es": a_no_es, "with_es": a_es},
}
print(results["dl"])


# -----------------------------------------------------------------------------
# 5) YOLO 파트 (라벨/평가/분석 예제 실행)
# -----------------------------------------------------------------------------
section("5) YOLO 파트")


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [xc, yc, w, h]

bbox_example = xyxy_to_yolo(50, 30, 180, 140, 640, 480)

# 간단 precision/recall 계산 예시
tp, fp, fn = 42, 8, 10
precision = tp / (tp + fp)
recall = tp / (tp + fn)

results["yolo"] = {
    "bbox_xyxy_to_yolo_example": [round(x, 6) for x in bbox_example],
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "note": "환경 제약으로 실제 YOLO 대규모 학습은 제외하고, 라벨/지표/분석 실습을 실행",
}
print(results["yolo"])


# -----------------------------------------------------------------------------
# 6) 프로젝트 템플릿 출력
# -----------------------------------------------------------------------------
section("6) 프로젝트 템플릿")
project_template = {
    "title": "CIFAR-100 분류 성능 개선",
    "target": "baseline 대비 +10%p",
    "required": ["실험로그", "재현 가능한 코드", "결과 해석"],
    "rubric": {
        "문제정의": 15,
        "데이터/전처리": 20,
        "모델링/실험": 25,
        "결과해석": 20,
        "문서/발표": 20,
    },
}
results["project"] = project_template
print(project_template)


# Save artifacts
with open(os.path.join(OUT_DIR, "all_examples_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n완료: outputs/all_examples_results.json 저장")
