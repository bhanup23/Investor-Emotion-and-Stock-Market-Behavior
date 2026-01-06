import os
import sys

PROJECT_ROOT = os.getcwd()

REQUIRED_PATHS = {
    "src": [
        "src/step3_emotion_extraction.py",
        "src/step4_aggregate_emotions.py",
        "src/step6A_panel_regression.py",
    ],
    "data": [
        "data/firm_day_emotions.csv",
        "data/firm_day_returns.csv",
    ],
    "results": [
        "results/panel_regression_results.csv",
        "results/synthetic_panel_results.csv",
    ],
    "root": [
        "README.md",
        "requirements.txt",
    ],
}

def check_paths():
    print("\n🔍 CHECKING PROJECT STRUCTURE\n" + "-" * 40)
    errors = 0

    for section, paths in REQUIRED_PATHS.items():
        print(f"\n📁 {section.upper()}")
        for path in paths:
            full_path = os.path.join(PROJECT_ROOT, path)
            if os.path.exists(full_path):
                print(f"  ✅ {path}")
            else:
                print(f"  ❌ MISSING: {path}")
                errors += 1

    if errors > 0:
        print(f"\n❌ STRUCTURE CHECK FAILED ({errors} issues found)")
        sys.exit(1)
    else:
        print("\n✅ ALL FILE PATHS VERIFIED")
        print("🚀 PROJECT IS REPRODUCIBLE AND READY")

if __name__ == "__main__":
    check_paths()
