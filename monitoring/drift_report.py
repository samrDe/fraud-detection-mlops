"""Create an Evidently data-drift report from reference and current feature data."""
import argparse
from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", default="reports/drift_report.html")
    args = parser.parse_args()
    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    if list(reference.columns) != list(current.columns):
        raise ValueError("Reference and current datasets must have identical columns in identical order.")
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(output))
    print(f"Drift report saved to {output}")

if __name__ == "__main__":
    main()
