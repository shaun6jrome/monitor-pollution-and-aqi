"""Tests for innovation benchmark metrics export to CSV and Markdown."""

from src.intelligence.innovation_evaluator import export_benchmark_csv, export_benchmark_markdown


def test_export_benchmark_csv_and_markdown(tmp_path):
    mock_results = {
        "comparison_table": [
            {
                "Method": "Baseline 1",
                "Discovered Events": 10,
                "True Events": 4,
                "Pairwise Precision": 0.5,
                "Pairwise Recall": 1.0,
                "Pairwise F1": 0.6667,
                "Primary Defect": "Overcounting",
            },
            {
                "Method": "Proposed",
                "Discovered Events": 4,
                "True Events": 4,
                "Pairwise Precision": 1.0,
                "Pairwise Recall": 1.0,
                "Pairwise F1": 1.0,
                "Primary Defect": "None",
            },
        ]
    }

    csv_file = tmp_path / "benchmark.csv"
    res_path = export_benchmark_csv(mock_results, output_path=csv_file)
    assert res_path.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "Baseline 1" in content
    assert "Proposed" in content

    md_table = export_benchmark_markdown(mock_results)
    assert "| Baseline 1 |" in md_table
    assert "| Proposed |" in md_table
    assert "Pairwise F1" in md_table
