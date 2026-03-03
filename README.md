# python-study

파이썬 + 머신러닝/딥러닝 예제를 실행하고, 결과를 **시각화 PNG**로 저장하는 프로젝트입니다.

## 실행

```powershell
python run_python_programming.py
python run_all_ml_dl_examples.py
```

인터프리터 충돌 시:

```powershell
& "C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe" run_python_programming.py
& "C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe" run_all_ml_dl_examples.py
```

## 생성되는 시각화 파일

- `charts/student_scores.png`
- `charts/sales_by_region.png`
- `charts/log_level_distribution.png`
- `charts/ml_logistic_confusion_matrix.png`
- `charts/ml_regression_rmse_comparison.png`
- `charts/ml_feature_importance_top5.png`
- `charts/dl_regularization_sweep.png`

## 결과 JSON

- `python-outputs/python_results.json`
- `outputs/all_examples_results.json`

## 시각화 스크린샷 (코드 + 결과)

### 1) Student Scores

```python
# 학생 점수 데이터
students = {
    "kim": 88,
    "lee": 92,
    "park": 76,
    "choi": 81,
}

# 시각화 저장
plt.figure(figsize=(8, 4.5))
plt.bar(list(students.keys()), list(students.values()))
plt.title("Student Scores")
plt.xlabel("Student")
plt.ylabel("Score")
plt.ylim(0, 100)
for i, v in enumerate(students.values()):
    plt.text(i, v + 1, str(v), ha="center")
plt.savefig("charts/student_scores.png", dpi=150)
plt.close()
```

![Student Scores](charts/student_scores.png)

### 2) Sales by Region

```python
# 지역별 매출 집계
sales_df = pd.DataFrame(
    {
        "region": ["Seoul", "Seoul", "Busan", "Busan", "Incheon", "Seoul"],
        "product": ["A", "B", "A", "C", "A", "C"],
        "amount": [120, 80, 100, 90, 70, 110],
    }
)
region_sum = sales_df.groupby("region")["amount"].sum().sort_values(ascending=False)

# 시각화 저장
plt.figure(figsize=(8, 4.5))
region_sum.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.savefig("charts/sales_by_region.png", dpi=150)
plt.close()
```

![Sales by Region](charts/sales_by_region.png)

### 3) Log Level Distribution

```python
# 로그 레벨 집계
logs = [
    "INFO login user=kim",
    "ERROR db timeout",
    "INFO view page=home",
    "WARNING cpu high",
    "ERROR db timeout",
    "INFO logout user=kim",
    "ERROR api 500",
    "INFO login user=lee",
]
level_counter = Counter(line.split()[0] for line in logs)

# 시각화 저장
plt.figure(figsize=(6, 6))
plt.pie(level_counter.values(), labels=level_counter.keys(), autopct="%1.1f%%", startangle=90)
plt.title("Log Level Distribution")
plt.savefig("charts/log_level_distribution.png", dpi=150)
plt.close()
```

![Log Level Distribution](charts/log_level_distribution.png)

### 4) ML Logistic Confusion Matrix

```python
# Logistic Regression 학습/예측
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
cm = confusion_matrix(y_test, pred)

# 시각화 저장
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Iris Logistic Confusion Matrix")
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.savefig("charts/ml_logistic_confusion_matrix.png", dpi=150)
plt.close()
```

![ML Logistic Confusion Matrix](charts/ml_logistic_confusion_matrix.png)

### 5) Regression RMSE Comparison

```python
# 회귀 모델 성능 계산
lin = LinearRegression().fit(Xr_train, yr_train)
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(Xr_train, yr_train)
lin_pred = lin.predict(Xr_test)
rf_pred = rf.predict(Xr_test)

rmse_vals = [
    math.sqrt(mean_squared_error(yr_test, lin_pred)),
    math.sqrt(mean_squared_error(yr_test, rf_pred)),
]

# 시각화 저장
plt.figure(figsize=(6, 4))
plt.bar(["LinearRegression", "RandomForest"], rmse_vals)
plt.title("Regression RMSE Comparison")
plt.ylabel("RMSE")
plt.savefig("charts/ml_regression_rmse_comparison.png", dpi=150)
plt.close()
```

![Regression RMSE Comparison](charts/ml_regression_rmse_comparison.png)

### 6) Feature Importance Top5

```python
# 중요도 계산
bc = load_breast_cancer()
Xb = pd.DataFrame(bc.data, columns=bc.feature_names)
yb = bc.target
rf_cls = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_cls.fit(Xb, yb)
imp = pd.Series(rf_cls.feature_importances_, index=Xb.columns).sort_values(ascending=False)

# 시각화 저장
plt.figure(figsize=(8, 4.5))
imp.head(5).sort_values().plot(kind="barh")
plt.title("Top-5 Feature Importance (Breast Cancer)")
plt.xlabel("Importance")
plt.savefig("charts/ml_feature_importance_top5.png", dpi=150)
plt.close()
```

![Feature Importance Top5](charts/ml_feature_importance_top5.png)

### 7) DL Regularization Sweep

```python
# alpha별 정확도 계산
reg_rows = []
for alpha in [0.0, 1e-4, 1e-3]:
    m = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=alpha, max_iter=25, random_state=42)),
    ])
    m.fit(Xd_train, yd_train)
    p = m.predict(Xd_test)
    reg_rows.append({"alpha": alpha, "acc": float(accuracy_score(yd_test, p))})

# 시각화 저장
plt.figure(figsize=(6, 4))
plt.plot([r["alpha"] for r in reg_rows], [r["acc"] for r in reg_rows], marker="o")
plt.title("Regularization Sweep (Digits MLP)")
plt.xlabel("alpha")
plt.ylabel("accuracy")
plt.xscale("symlog", linthresh=1e-6)
plt.savefig("charts/dl_regularization_sweep.png", dpi=150)
plt.close()
```

![DL Regularization Sweep](charts/dl_regularization_sweep.png)
