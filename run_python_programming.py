import json
import os
from collections import Counter

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "python-outputs")
os.makedirs(OUT_DIR, exist_ok=True)

summary = {}

print("=" * 70)
print("Day 1) Python 기초 문법/자료구조")
print("=" * 70)

# Day1-1: 리스트/딕셔너리/함수
scores = [88, 92, 76, 81, 95, 68, 90]
avg_score = sum(scores) / len(scores)
passed = [s for s in scores if s >= 80]

students = {
    "kim": 88,
    "lee": 92,
    "park": 76,
    "choi": 81,
}
top_student = max(students, key=students.get)


def grade(score):
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    return "D"


summary["day1"] = {
    "avg_score": round(avg_score, 2),
    "pass_count": len(passed),
    "top_student": top_student,
    "top_grade": grade(students[top_student]),
}
print(summary["day1"])

print("\n" + "=" * 70)
print("Day 2) NumPy/Pandas 데이터 처리")
print("=" * 70)

# Day2-1: NumPy 연산
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr_stats = {
    "shape": arr.shape,
    "sum": int(arr.sum()),
    "mean": float(arr.mean()),
    "col_mean": arr.mean(axis=0).round(2).tolist(),
}

# Day2-2: Pandas 집계
sales_df = pd.DataFrame(
    {
        "region": ["Seoul", "Seoul", "Busan", "Busan", "Incheon", "Seoul"],
        "product": ["A", "B", "A", "C", "A", "C"],
        "amount": [120, 80, 100, 90, 70, 110],
    }
)
region_sum = sales_df.groupby("region")["amount"].sum().sort_values(ascending=False)
product_sum = sales_df.groupby("product")["amount"].sum().sort_values(ascending=False)

summary["day2"] = {
    "numpy": arr_stats,
    "region_sum": region_sum.to_dict(),
    "product_sum": product_sum.to_dict(),
}
print(summary["day2"])

print("\n" + "=" * 70)
print("Day 3) 미니 프로젝트: 로그 분석기")
print("=" * 70)

# Day3-1: 텍스트 로그 분석
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

summary["day3"] = {
    "total_logs": len(logs),
    "level_count": dict(level_counter),
    "error_count": len(error_messages),
    "errors": error_messages,
}
print(summary["day3"])

# 학습자 과제 템플릿
assignments = {
    "day1": [
        "사용자 입력 점수 리스트를 받아 평균/최고점/최저점을 출력하는 함수 작성",
        "딕셔너리로 학생-점수 관리 후 A/B/C/D 등급 분포 계산",
    ],
    "day2": [
        "CSV를 읽어 결측치 개수 확인 후 평균 대체",
        "지역별 매출 합계/평균을 구하고 막대그래프로 시각화",
    ],
    "day3": [
        "로그 파일에서 ERROR 비율 계산",
        "가장 자주 발생한 에러 Top-3 추출",
        "분석 결과를 JSON 보고서로 저장",
    ],
}

summary["assignments"] = assignments

# 저장
result_path = os.path.join(OUT_DIR, "python_results.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n저장 완료:", result_path)
