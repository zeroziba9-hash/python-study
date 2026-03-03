from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).parent
CHARTS = BASE / "charts"
OUT = CHARTS / "with_code"
OUT.mkdir(parents=True, exist_ok=True)

items = [
    (
        "student_scores.png",
        "student_scores_with_code.png",
        """# Student Scores
plt.figure(figsize=(8, 4.5))
plt.bar(list(students.keys()), list(students.values()))
plt.title(\"Student Scores\")
plt.savefig(\"charts/student_scores.png\", dpi=150)
plt.close()""",
    ),
    (
        "sales_by_region.png",
        "sales_by_region_with_code.png",
        """# Sales by Region
plt.figure(figsize=(8, 4.5))
region_sum.plot(kind=\"bar\")
plt.title(\"Sales by Region\")
plt.savefig(\"charts/sales_by_region.png\", dpi=150)
plt.close()""",
    ),
    (
        "log_level_distribution.png",
        "log_level_distribution_with_code.png",
        """# Log Level Distribution
plt.figure(figsize=(6, 6))
plt.pie(level_counter.values(), labels=level_counter.keys(),
        autopct=\"%1.1f%%\", startangle=90)
plt.title(\"Log Level Distribution\")
plt.savefig(\"charts/log_level_distribution.png\", dpi=150)
plt.close()""",
    ),
    (
        "ml_logistic_confusion_matrix.png",
        "ml_logistic_confusion_matrix_with_code.png",
        """# ML Logistic Confusion Matrix
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap=\"Blues\")
plt.title(\"Iris Logistic Confusion Matrix\")
plt.colorbar()
plt.savefig(\"charts/ml_logistic_confusion_matrix.png\", dpi=150)
plt.close()""",
    ),
    (
        "ml_regression_rmse_comparison.png",
        "ml_regression_rmse_comparison_with_code.png",
        """# Regression RMSE Comparison
plt.figure(figsize=(6, 4))
plt.bar([\"LinearRegression\", \"RandomForest\"], rmse_vals)
plt.title(\"Regression RMSE Comparison\")
plt.ylabel(\"RMSE\")
plt.savefig(\"charts/ml_regression_rmse_comparison.png\", dpi=150)
plt.close()""",
    ),
    (
        "ml_feature_importance_top5.png",
        "ml_feature_importance_top5_with_code.png",
        """# Feature Importance Top5
plt.figure(figsize=(8, 4.5))
imp.head(5).sort_values().plot(kind=\"barh\")
plt.title(\"Top-5 Feature Importance (Breast Cancer)\")
plt.xlabel(\"Importance\")
plt.savefig(\"charts/ml_feature_importance_top5.png\", dpi=150)
plt.close()""",
    ),
    (
        "dl_regularization_sweep.png",
        "dl_regularization_sweep_with_code.png",
        """# DL Regularization Sweep
plt.figure(figsize=(6, 4))
plt.plot([r[\"alpha\"] for r in reg_rows],
         [r[\"acc\"] for r in reg_rows], marker=\"o\")
plt.title(\"Regularization Sweep (Digits MLP)\")
plt.xlabel(\"alpha\")
plt.ylabel(\"accuracy\")
plt.savefig(\"charts/dl_regularization_sweep.png\", dpi=150)
plt.close()""",
    ),
]

try:
    font = ImageFont.truetype("consola.ttf", 18)
except Exception:
    font = ImageFont.load_default()

for src_name, out_name, code in items:
    src = CHARTS / src_name
    if not src.exists():
        continue

    chart = Image.open(src).convert("RGB")
    code_lines = code.split("\n")
    line_h = 26
    padding = 24
    code_h = padding * 2 + line_h * len(code_lines)

    canvas_w = max(chart.width, 1200)
    canvas_h = code_h + chart.height + 20

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(12, 17, 23))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, canvas_w, code_h], fill=(20, 30, 40))

    y = padding
    for line in code_lines:
        draw.text((padding, y), line, fill=(220, 235, 255), font=font)
        y += line_h

    x_chart = (canvas_w - chart.width) // 2
    canvas.paste(chart, (x_chart, code_h))

    canvas.save(OUT / out_name)

print(f"Generated images in: {OUT}")
