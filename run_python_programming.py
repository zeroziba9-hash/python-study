import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "python-outputs")
CHART_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

summary = {}

print("=" * 70)
print("Python 기초/데이터 처리 실습")
print("=" * 70)

scores = [88, 92, 76, 81, 95, 68, 90]
avg_score = sum(scores) / len(scores)
passed = [s for s in scores if s >= 80]

students = {"kim": 88, "lee": 92, "park": 76, "choi": 81}

summary["python_basics"] = {
    "avg_score": round(avg_score, 2),
    "pass_count": len(passed),
    "top_student": max(students, key=students.get),
}
print(summary["python_basics"])

# 시각화 1: 학생 점수
plt.figure(figsize=(8, 4.5))
plt.bar(list(students.keys()), list(students.values()))
plt.title("Student Scores")
plt.xlabel("Student")
plt.ylabel("Score")
plt.ylim(0, 100)
for i, v in enumerate(students.values()):
    plt.text(i, v + 1, str(v), ha="center")
score_chart = os.path.join(CHART_DIR, "student_scores.png")
plt.tight_layout()
plt.savefig(score_chart, dpi=150)
plt.close()

# 데이터 처리
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr_stats = {
    "shape": arr.shape,
    "sum": int(arr.sum()),
    "mean": float(arr.mean()),
    "col_mean": arr.mean(axis=0).round(2).tolist(),
}

sales_df = pd.DataFrame(
    {
        "region": ["Seoul", "Seoul", "Busan", "Busan", "Incheon", "Seoul"],
        "product": ["A", "B", "A", "C", "A", "C"],
        "amount": [120, 80, 100, 90, 70, 110],
    }
)
region_sum = sales_df.groupby("region")["amount"].sum().sort_values(ascending=False)
product_sum = sales_df.groupby("product")["amount"].sum().sort_values(ascending=False)

summary["data_processing"] = {
    "numpy": arr_stats,
    "region_sum": region_sum.to_dict(),
    "product_sum": product_sum.to_dict(),
}
print(summary["data_processing"])

# 시각화 2: 지역별 매출
plt.figure(figsize=(8, 4.5))
region_sum.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.xticks(rotation=0)
region_chart = os.path.join(CHART_DIR, "sales_by_region.png")
plt.tight_layout()
plt.savefig(region_chart, dpi=150)
plt.close()

# 로그 분석
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
error_messages = [line for line in logs if line.startswith("ERROR")]

summary["log_analysis"] = {
    "total_logs": len(logs),
    "level_count": dict(level_counter),
    "error_count": len(error_messages),
    "errors": error_messages,
}
print(summary["log_analysis"])

# 시각화 3: 로그 레벨 비율
plt.figure(figsize=(6, 6))
plt.pie(level_counter.values(), labels=level_counter.keys(), autopct="%1.1f%%", startangle=90)
plt.title("Log Level Distribution")
log_chart = os.path.join(CHART_DIR, "log_level_distribution.png")
plt.tight_layout()
plt.savefig(log_chart, dpi=150)
plt.close()

summary["charts"] = {
    "student_scores": os.path.relpath(score_chart, BASE_DIR),
    "sales_by_region": os.path.relpath(region_chart, BASE_DIR),
    "log_level_distribution": os.path.relpath(log_chart, BASE_DIR),
}

result_path = os.path.join(OUT_DIR, "python_results.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n저장 완료:", result_path)
print("시각화 저장:", summary["charts"])
