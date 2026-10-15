"""원자료 감사의 행동 집계와 과학 그림을 생성한다. 실험 조건을 수정하지 않는다."""

import argparse
from collections import Counter
from pathlib import Path

from evosim_interaction.storage import read_json, write_json


def summarize_behavior(root):
    root = Path(root)
    audit = read_json(root / "audit.json")
    result = read_json(root / "summary.json")
    relations = {label: Counter() for label in ("left", "right", "at_goal")}
    counts, changed = Counter(), Counter()
    episodes = repeating = 0
    for key, value in audit["behavior"].items():
        if key.startswith("evolution_"):
            episodes += value["episodes"]
            repeating += value["failures_with_last_six_identical_actions"]
            counts.update(value["goals_collected"])
            changed.update(value["after_goal_change"])
            for label, relation in value["relations"].items():
                relations[label].update(relation)
    behavioral = {
        "episodes": episodes, "goals_collected": dict(counts),
        "goal_count_percent": {str(i): counts[str(i)] / episodes * 100 for i in range(4)},
        "relations": {label: {**dict(value), "correct_fraction": value["correct"] / value["steps"]}
                      for label, value in relations.items()},
        "after_goal_change": {**dict(changed), "correct_fraction": changed["correct"] / changed["steps"]},
        "failures_with_last_six_identical_actions": repeating,
        "limitation": "Pooled visited-state counts, not independent trials or a causal decomposition",
        "mean_evolution_collected": sum(row["evolution"]["mean_collected"] for row in result["rows"]) / 10,
        "mean_random_search_collected": sum(row["random"]["mean_collected"] for row in result["rows"]) / 10,
        "audit_counts": audit["counts"],
    }
    write_json(root / "behavior_summary.json", behavioral)
    print(behavioral)
    from reportlab.graphics import renderSVG
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font = "Malgun Gothic"
    pdfmetrics.registerFont(TTFont(font, "C:/Windows/Fonts/malgun.ttf"))
    drawing = Drawing(1000, 430)
    drawing.add(String(25, 402, "Q1 첫 실험: 부분 수집은 나타났지만 과제 완수는 드묾", fontName=font, fontSize=18))
    values = [result["evolution_mean"] * 100, result["random_mean"] * 100,
              result["uniform_random"]["success_rate"] * 100, result["blind_success_rate"] * 100]
    percentages = [counts[str(i)] / episodes * 100 for i in range(4)]
    for x, data, names, maximum, title in (
        (55, values, ["진화", "무작위 개체", "무작위 행동", "고정 순서"], 5.2, "세 목표 완료 성공률 (%)"),
        (555, percentages, ["0개", "1개", "2개", "3개"], 70, "진화 후보의 수집 개수별 에피소드 비율 (%)"),
    ):
        chart = VerticalBarChart()
        chart.x, chart.y, chart.width, chart.height = x, 95, 400, 240
        chart.data = [data]
        chart.categoryAxis.categoryNames = names
        chart.categoryAxis.labels.fontName = font
        chart.categoryAxis.labels.fontSize = 10
        chart.valueAxis.valueMin, chart.valueAxis.valueMax = 0, maximum
        chart.valueAxis.valueStep = 1 if maximum < 10 else 10
        chart.valueAxis.labels.fontName = font
        chart.bars[0].fillColor = colors.HexColor("#2563eb")
        chart.barLabelFormat = "%.2f%%"
        chart.barLabels.fontName = font
        chart.barLabels.fontSize = 10
        chart.barLabels.nudge = 7
        drawing.add(chart)
        drawing.add(String(x, 355, title, fontName=font, fontSize=12))
    drawing.add(String(30, 45, "좌표 비교 규칙과 수작업 신경망은 100% 해결. 수작업 신경망은 진화로 발견한 개체가 아님.", fontName=font, fontSize=11))
    drawing.add(String(30, 22, "진화·무작위 탐색은 10개 시드 평균. 우측은 진화 후보의 504개 평가 × 3반복 × 10시드 기록.", fontName=font, fontSize=11))
    renderSVG.drawToFile(drawing, str(root / "q1_results.svg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    summarize_behavior(parser.parse_args().root)
