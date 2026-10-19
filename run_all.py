"""
Orchestrator for the Air Quality Image Classification pipeline.

Runs the full, scientifically-ordered workflow, or a single stage:

    python run_all.py --stage audit      # env + discovery + audit + EDA + split
    python run_all.py --stage screen     # short candidate screening
    python run_all.py --stage train      # full training of selected candidates
    python run_all.py --stage evaluate   # ONE-shot test evaluation
    python run_all.py --stage explain    # error analysis + Grad-CAM
    python run_all.py --stage all        # everything above, in order
    python run_all.py --stage app        # launch the dashboard

Data-prep stages (audit) have no heavy dependencies. Training/evaluation stages
require TensorFlow (see requirements.txt). Each stage is safe to re-run.
"""
from __future__ import annotations

import argparse
import sys

from src import config, utils

log = utils.get_logger("run_all")


def stage_audit() -> None:
    from src import env_audit, dataset_analysis, splits
    env_audit.run_environment_audit()
    df = None
    # discovery/build happens inside each call; do audit -> labels -> eda -> split.
    dataset_analysis.audit_dataset()
    dataset_analysis.verify_labels()
    dataset_analysis.run_eda()
    manifest = splits.make_split()
    if not manifest["zero_leakage"]:
        log.warning("Split reported non-zero group overlap — inspect manifest!")


def stage_screen() -> None:
    from src import screening
    screening.screen()
    screening.select_candidates()


def stage_train() -> None:
    from src import screening
    chosen = screening.select_candidates()
    screening.full_train(chosen)
    screening.select_final_model()


def stage_evaluate() -> None:
    from src import evaluate
    evaluate.evaluate_final()


def stage_explain() -> None:
    from src import error_analysis, explainability
    error_analysis.analyze()
    explainability.generate_test_examples()


def stage_app() -> None:
    from app.app import app
    config.ensure_dirs()
    log.info("Launching dashboard on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


STAGES = {
    "audit": stage_audit,
    "screen": stage_screen,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "explain": stage_explain,
    "app": stage_app,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Air Quality DL pipeline runner.")
    ap.add_argument("--stage", default="all",
                    choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    config.ensure_dirs()

    if args.stage == "all":
        for name in ("audit", "screen", "train", "evaluate", "explain"):
            log.info("==== STAGE: %s ====", name.upper())
            STAGES[name]()
        log.info("Pipeline complete. Launch the UI with: python run_all.py --stage app")
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    sys.exit(main())
