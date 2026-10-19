"""
Smart model screening + final selection.

To avoid wasting compute we DO NOT fully train every architecture. Instead:
  1. screen()               - short, frozen-backbone runs on ALL candidates
                              (same split, same protocol) -> cheap comparison.
  2. select_candidates()    - keep the top-K by validation Macro-F1 with an
                              efficiency-aware tie-break.
  3. full_train()           - full transfer-learning + fine-tuning only on those.
  4. select_final_model()   - documented rule over the FULL runs (never uses
                              the test set).
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from . import config, utils
from . import train as trainer

log = utils.get_logger("screening")


def _apply_env_recommendations(cfg) -> None:
    env = utils.load_json(config.ENV_AUDIT_JSON, default={})
    rec = env.get("recommended", {}) if env else {}
    cfg.mixed_precision = bool(rec.get("mixed_precision", False))
    if rec.get("suggested_batch_size"):
        cfg.batch_size = rec["suggested_batch_size"]


def screen(candidates: List[str] | None = None, epochs: int = 4,
           image_size: int = config.IMAGE_SIZE) -> pd.DataFrame:
    """Short frozen-backbone screening across candidate architectures."""
    candidates = candidates or config.CANDIDATE_MODELS
    rows = []
    for name in candidates:
        cfg = config.TrainConfig(model_name=name, image_size=image_size,
                                 head_epochs=epochs, finetune_epochs=0,
                                 early_stop_patience=epochs)
        _apply_env_recommendations(cfg)
        try:
            res = trainer.train_model(cfg, experiment_id=f"screen_{name}",
                                      notes="screening", screening=True)
            rows.append(res)
        except Exception as exc:  # keep screening the rest
            log.warning("Screening failed for %s: %s", name, exc)
    df = pd.DataFrame(rows)
    log.info("Screening complete for %d candidates", len(df))
    return df


def select_candidates(top_k: int = 2) -> List[str]:
    """Pick the strongest screened models (Macro-F1, efficiency tie-break)."""
    df = pd.read_csv(config.EXPERIMENTS_CSV)
    screened = df[df["notes"] == "screening"].copy()
    if screened.empty:
        return config.CANDIDATE_MODELS[:top_k]
    # Rank: higher Macro-F1 first; break ties toward smaller/faster models.
    screened["eff_rank"] = (
        screened["val_macro_f1"].rank(ascending=False) * 1.0
        + screened["model_size_mb"].rank(ascending=True) * 0.15
        + screened["training_time_sec"].rank(ascending=True) * 0.10
    )
    chosen = (screened.sort_values("eff_rank")
              .head(top_k)["model"].tolist())
    log.info("Selected candidates for full training: %s", chosen)
    return chosen


def full_train(candidates: List[str]) -> pd.DataFrame:
    """Full transfer learning + fine-tuning on the selected candidates."""
    rows = []
    for name in candidates:
        cfg = config.TrainConfig(model_name=name)
        _apply_env_recommendations(cfg)
        res = trainer.train_model(cfg, experiment_id=f"full_{name}",
                                  notes="full")
        rows.append(res)
    return pd.DataFrame(rows)


def select_final_model() -> Dict[str, Any]:
    """Documented selection rule over FULL runs. Never touches the test set."""
    df = pd.read_csv(config.EXPERIMENTS_CSV)
    full = df[df["notes"] == "full"].copy()
    if full.empty:
        raise RuntimeError("No full training runs found to select from.")

    # Primary: validation Macro-F1. Secondary: validation accuracy.
    # Tertiary (tie-break within 0.01 F1): prefer smaller model (efficiency).
    full = full.sort_values(["val_macro_f1", "val_accuracy"], ascending=False)
    best_f1 = full.iloc[0]["val_macro_f1"]
    contenders = full[full["val_macro_f1"] >= best_f1 - 0.01]
    winner = contenders.sort_values("model_size_mb").iloc[0]

    selection = {
        "selected_model": winner["model"],
        "selected_experiment": winner["experiment"],
        "selection_rule": ("Highest validation Macro-F1; ties within 0.01 F1 "
                           "broken toward the smaller model for efficiency. "
                           "Test set NOT used in selection."),
        "val_accuracy": float(winner["val_accuracy"]),
        "val_macro_f1": float(winner["val_macro_f1"]),
        "parameters": int(winner["parameters"]),
        "model_size_mb": float(winner["model_size_mb"]),
        "image_size": int(winner["image_size"]),
        "model_path": str(config.MODELS_DIR /
                          f"{winner['experiment']}.keras"),
        "considered": contenders[["model", "val_macro_f1", "val_accuracy",
                                  "model_size_mb"]].to_dict("records"),
    }
    utils.save_json(config.MODEL_SELECTION_JSON, selection)
    utils.save_json(config.FINAL_MODEL_META_JSON, selection)
    log.info("FINAL MODEL: %s (val Macro-F1=%.4f, %.1f MB)",
             selection["selected_model"], selection["val_macro_f1"],
             selection["model_size_mb"])
    return selection


if __name__ == "__main__":
    screen()
    chosen = select_candidates()
    full_train(chosen)
    select_final_model()
