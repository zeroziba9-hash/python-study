# python-study

파이썬 기초 실습 + 머신러닝/딥러닝 예제 실행 프로젝트입니다.

## 1) 프로젝트 목적

- 파이썬 기본 문법/자료구조/데이터 처리 실습
- 머신러닝 기본 모델 학습 및 평가 흐름 익히기
- 딥러닝(MLP 기반) 실험, 정규화/조기종료 개념 체험
- YOLO 파트는 환경 제약을 고려해 라벨 변환/평가지표 중심으로 실습

---

## 2) 폴더 구조

```text
python-study/
├─ run_python_programming.py
├─ run_all_ml_dl_examples.py
├─ python-outputs/
│  └─ python_results.json
├─ outputs/
│  └─ all_examples_results.json
├─ charts/
│  ├─ day1_student_scores.png
│  ├─ day2_sales_by_region.png
│  ├─ day3_log_level_distribution.png
│  ├─ ml_logistic_confusion_matrix.png
│  ├─ ml_regression_rmse_comparison.png
│  ├─ ml_feature_importance_top5.png
│  └─ dl_regularization_sweep.png
└─ README.md
```

---

## 3) 실행 방법

현재 폴더(`python-study`)에서 터미널 실행:

```powershell
python run_python_programming.py
python run_all_ml_dl_examples.py
```

### 인터프리터 문제 시 (권장: Python 3.10 직접 지정)

```powershell
& "C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe" run_python_programming.py
& "C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe" run_all_ml_dl_examples.py
```

### 시각화 파일 생성

두 스크립트를 실행하면 `charts/`에 PNG가 자동 생성됩니다.

---

## 4) 주요 코드 + 결과 예시

## A. `run_python_programming.py`

### 핵심 코드 예시 (Day2: Pandas 집계)

```python
sales_df = pd.DataFrame(
    {
        "region": ["Seoul", "Seoul", "Busan", "Busan", "Incheon", "Seoul"],
        "product": ["A", "B", "A", "C", "A", "C"],
        "amount": [120, 80, 100, 90, 70, 110],
    }
)
region_sum = sales_df.groupby("region")["amount"].sum().sort_values(ascending=False)
product_sum = sales_df.groupby("product")["amount"].sum().sort_values(ascending=False)
```

### 실행 결과 예시

```text
Day 1) Python 기초 문법/자료구조
{'avg_score': 84.29, 'pass_count': 5, 'top_student': 'lee', 'top_grade': 'A'}

Day 2) NumPy/Pandas 데이터 처리
{'numpy': {'shape': (2, 3), 'sum': 21, 'mean': 3.5, 'col_mean': [2.5, 3.5, 4.5]},
 'region_sum': {'Seoul': 310, 'Busan': 190, 'Incheon': 70},
 'product_sum': {'A': 290, 'C': 200, 'B': 80}}

Day 3) 미니 프로젝트: 로그 분석기
{'total_logs': 8, 'level_count': {'INFO': 4, 'ERROR': 3, 'WARNING': 1},
 'error_count': 3,
 'errors': ['ERROR db timeout', 'ERROR db timeout', 'ERROR api 500']}
```

출력 파일:
- `python-outputs/python_results.json`

---

## B. `run_all_ml_dl_examples.py`

### 핵심 코드 예시 (ML: Logistic Regression)

```python
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
print("Logistic Regression accuracy:", accuracy_score(y_test, pred))
```

### 실행 결과 예시 (요약)

```text
Logistic Regression accuracy: 0.9333333333333333

SVM GridSearch:
{'best_params': {'svm__C': 10, 'svm__gamma': 'scale', 'svm__kernel': 'rbf'},
 'best_cv': 0.993103448275862,
 'test_acc': 0.9444444444444444}

Digits MLP accuracy: 0.9638888888888889

YOLO 파트:
{'bbox_xyxy_to_yolo_example': [0.179688, 0.177083, 0.203125, 0.229167],
 'precision': 0.84,
 'recall': 0.8077}
```

출력 파일:
- `outputs/all_examples_results.json`

---

## 5) 결과 파일

### `python-outputs/python_results.json`
- Day1/Day2/Day3 실습 결과 JSON

### `outputs/all_examples_results.json`
- Intro/준비/개념 + ML/DL/YOLO + 프로젝트 템플릿 결과 JSON

---

## 6) 자주 발생하는 문제

### Q1. `ModuleNotFoundError: numpy/pandas/sklearn`
A. 현재 인터프리터가 패키지 설치된 Python과 다를 가능성이 큽니다.

- VSCode에서 인터프리터를 Python 3.10으로 선택하거나,
- 위의 "Python 3.10 직접 지정" 명령으로 실행하세요.

### Q2. `ConvergenceWarning`이 뜹니다.
A. 오류가 아니라 학습 반복 횟수 관련 경고입니다. 스크립트는 정상 실행됩니다.
필요하면 `max_iter`를 늘려 경고를 줄일 수 있습니다.

---

## 7) 다음 확장 아이디어

- PyTorch 기반 CNN 학습 파이프라인 추가
- Ultralytics YOLO 실제 학습/검증 스크립트 추가
- 실험 로그(CSV/MLflow) 자동 저장 기능 추가
- CLI 옵션(`--epochs`, `--seed`) 지원
