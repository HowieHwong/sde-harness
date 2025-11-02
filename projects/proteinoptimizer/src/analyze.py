from __future__ import annotations
import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import math

# ------------------------------
# Aliases / parsing
# ------------------------------
DATASET_ALIASES = {
    "aav": "AAV",
    "gb1": "GB1",
    "gfp": "GFP",
    "trpb": "TrpB",
    "syn-3bfo": "Syn-3bfo",
    "syn3bfo": "Syn-3bfo",
    "3bfo": "Syn-3bfo",
}

MODEL_ALIASES = {
    "baseline": "baseline",
    "gpt-5-mini": "gpt-5-mini",
    "gpt5mini": "gpt-5-mini",
    "gpt": "gpt",
}

TASK_ALIASES = {
    "single": "single",
    "multi": "multi",
    "multisite": "multi",
    "unspecified": "unspecified",
}

@dataclass
class ParsedName:
    dataset: str
    task: str
    model: str
    run_id: Optional[str] = None

def _normalize_token(token: str) -> str:
    return token.strip().lower().replace(" ", "").replace("__", "_")

def parse_filename(path: str) -> ParsedName:
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    if stem.startswith("results_"):
        stem = stem[len("results_"):]
    tokens = [_normalize_token(t) for t in stem.split("_")]
    task = "unspecified"
    dataset = "unknown"
    model = "unknown"
    run_id: Optional[str] = None
    for i, tok in enumerate(tokens):
        if tok in TASK_ALIASES:
            task = TASK_ALIASES[tok]
            if i + 1 < len(tokens):
                dataset = tokens[i + 1]
            if len(tokens) >= 3:
                model = tokens[-1]
            break
    else:
        if tokens:
            dataset = tokens[0]
        if len(tokens) >= 2:
            model = tokens[-1]
    for tok in tokens:
        if re.fullmatch(r"\d+", tok):
            run_id = tok
            break
    dataset = DATASET_ALIASES.get(dataset, dataset.upper())
    model = MODEL_ALIASES.get(model, model)
    task = TASK_ALIASES.get(task, task)
    return ParsedName(dataset=dataset, task=task, model=model, run_id=run_id)

# ------------------------------
# Score extraction / Top-K
# ------------------------------
def extract_best_score(payload: Dict, higher_is_better: bool) -> Tuple[Optional[float], Optional[str]]:
    """
    Best single score/sequence. If not present, derive from final_population.
    """
    best_score = payload.get("best_score", None)
    best_seq = payload.get("best_sequence", None)
    if best_score is None or best_seq is None:
        final_pop = payload.get("final_population", None)
        if isinstance(final_pop, list) and final_pop and isinstance(final_pop[0], (list, tuple)):
            if higher_is_better:
                seq, score = max(final_pop, key=lambda x: x[1])
            else:
                seq, score = min(final_pop, key=lambda x: x[1])
            best_seq = seq
            best_score = float(score)
    return (None if best_score is None else float(best_score),
            None if best_seq is None else str(best_seq))

def _collect_scores_from_payload(payload: Dict[str, Any]) -> List[float]:
    """
    Collect raw candidate scores from payload.
    Prefers final_population; falls back to all_results.
    """
    scores: List[float] = []
    pop = payload.get("final_population")
    if isinstance(pop, list) and pop:
        for item in pop:
            # Expect [sequence, score]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    scores.append(float(item[1]))
                except Exception:
                    continue
    if not scores:
        all_res = payload.get("all_results")
        if isinstance(all_res, dict) and all_res:
            for v in all_res.values():
                try:
                    scores.append(float(v))
                except Exception:
                    continue
    return scores

def compute_topk(scores: List[float], higher_is_better: bool, k: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (kth_score_at_rank_k, mean_of_top_k), where 'top' respects higher_is_better.
    If there are fewer than k items, returns (None, None).
    """
    if not isinstance(scores, list) or len(scores) < k:
        return (None, None)
    srt = sorted(scores, reverse=higher_is_better)
    kth = float(srt[k-1])
    mean_k = float(sum(srt[:k]) / k)
    return (kth, mean_k)

def load_file(path: str, higher_is_better: bool) -> Dict:
    with open(path, "r") as f:
        payload = json.load(f)

    parsed = parse_filename(path)
    best_score, best_seq = extract_best_score(payload, higher_is_better)

    # History
    hist = payload.get("best_scores_history", [])
    history_len = len(hist)
    history_final = float(hist[-1]) if history_len > 0 else None
    history_best = (max(hist) if higher_is_better else min(hist)) if history_len > 0 else None

    # Population + Top-K
    pop = payload.get("final_population", [])
    pop_size = len(pop) if isinstance(pop, list) else 0
    all_scores = _collect_scores_from_payload(payload)

    top1_score = best_score  # by definition of "best" this is the top-1 score
    # Still compute defensively from raw scores (in case best_score missing)
    if top1_score is None and all_scores:
        srt = sorted(all_scores, reverse=higher_is_better)
        top1_score = float(srt[0])

    top10_kth, top10_mean = compute_topk(all_scores, higher_is_better, k=10)
    top5_kth, top5_mean = compute_topk(all_scores, higher_is_better, k=5)

    rec = {
        "file": path,
        "dataset": parsed.dataset,
        "task": parsed.task,
        "model": parsed.model,
        "run_id": parsed.run_id,

        "best_score": best_score,
        "best_sequence": best_seq,

        "history_len": history_len,
        "history_final": history_final,
        "history_best": history_best,

        "population_size": pop_size,

        # Top-K metrics
        # 'top1_score' is the best (rank-1) score
        "top1_score": top1_score,
        # 'top10_score'/'top5_score' are the scores at rank 10/5 respectively
        "top10_score": top10_kth,
        "top5_score": top5_kth,
        # 'mean_top10'/'mean_top5' are the average score over the top 10/5 candidates
        "mean_top10": top10_mean,
        "mean_top5": top5_mean,
    }
    return rec

# ------------------------------
# Main: print ONE table only
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Summarize results with a single Top-1/Top-5/Top-10 table for Baseline and GPT.")
    ap.add_argument("--glob", type=str, default="results/*.json",
                    help="Glob pattern for result JSON files (e.g., 'projects/proteinoptimizer/results/*.json').")
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() in ('1','true','yes','y'),
                    default=False, help="True if larger scores are better (default: False, lower is better).")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"No files matched: {args.glob}")
        return

    rows = []
    for fpath in files:
        try:
            rec = load_file(fpath, higher_is_better=args.higher_is_better)
            rows.append(rec)
        except Exception as e:
            print(f"[WARN] Failed to parse {fpath}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid records parsed.")
        return

    # Collapse model names into two families: Baseline, GPT (anything starting with 'gpt')
    def _family(m: Any) -> str:
        m = (m or "")
        m = str(m).lower()
        if m.startswith("gpt"):
            return "GPT"
        if m == "baseline":
            return "Baseline"
        return "Other"

    df["family"] = df["model"].apply(_family)

    # Keep only Baseline and GPT rows
    df = df[df["family"].isin(["Baseline", "GPT"])].copy()
    if df.empty:
        print("No Baseline/GPT records found in parsed files.")
        return

    # Compute OVERALL means for each family
    # Define Top-1 as mean of top1_score
    # Define Top-5 as mean of mean_top5 (average of each run's top-5 scores)
    # Define Top-10 as mean of mean_top10 (average of each run's top-10 scores)
    agg = (
        df.groupby("family", dropna=False)
          .agg(
              Top_1=("top1_score", "mean"),
              Top_5=("mean_top5", "mean"),
              Top_10=("mean_top10", "mean"),
              N=("file", "count"),
          )
          .reset_index()
    )

    # Order rows: Baseline first, then GPT
    order = pd.CategoricalDtype(categories=["Baseline", "GPT"], ordered=True)
    agg["family"] = agg["family"].astype(order)
    agg = agg.sort_values("family")

    # Round for display
    display = agg[["family", "Top_1", "Top_5", "Top_10"]].copy()
    display = display.rename(columns={"family": "Model"})
    display = display.round({"Top_1": 4, "Top_5": 4, "Top_10": 4})

    # Print exactly one table
    try:
        print(display.to_markdown(index=False))
    except Exception:
        # Fallback if tabulate isn't available
        print(display.to_string(index=False))

if __name__ == "__main__":
    main()
