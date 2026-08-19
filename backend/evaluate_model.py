"""
evaluate_model.py
------------------
Runs a proper evaluation of your trained YOLO model on the validation/test
split and prints/save the metrics you actually need for your README and
LinkedIn post: precision, recall, mAP@50, mAP@50:95, per-class breakdown,
confusion matrix, and inference latency/FPS on your own hardware.

This replaces "eyeballing one screenshot" with numbers you can defend.

Usage:
    python evaluate_model.py --weights runs/detect/train/weights/best.pt \
                              --data road_damage.yaml \
                              --split val

Install deps:
    pip install ultralytics
"""

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def run_evaluation(weights: str, data_yaml: str, split: str = "val", imgsz: int = 640):
    model = YOLO(weights)

    print(f"\nEvaluating {weights} on split='{split}' using {data_yaml}\n" + "-" * 60)
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, plots=True)

    # Ultralytics' `metrics.box` gives overall + per-class arrays.
    overall = {
        "precision": float(metrics.box.mp),        # mean precision across classes
        "recall": float(metrics.box.mr),            # mean recall across classes
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    class_names = metrics.names
    per_class = {}
    # metrics.box.ap_class_index maps result rows to class indices
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        per_class[class_names[int(class_idx)]] = {
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "AP50": float(metrics.box.ap50[i]),
            "AP50_95": float(metrics.box.ap[i]),
        }

    print("\nOVERALL METRICS")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")

    print("\nPER-CLASS METRICS")
    for cls, vals in per_class.items():
        print(f"  {cls}:")
        for k, v in vals.items():
            print(f"    {k}: {v:.4f}")

    # Confusion matrix + PR/F1 curve plots are auto-saved by Ultralytics
    # (plots=True above) into the run's save_dir — point to that in your README.
    print(f"\nConfusion matrix & curves saved under: {metrics.save_dir}")

    return {"overall": overall, "per_class": per_class, "plots_dir": str(metrics.save_dir)}


def measure_latency(weights: str, sample_image: str, n_runs: int = 30, imgsz: int = 640):
    """
    Measures real inference latency/FPS on YOUR hardware — don't quote
    someone else's benchmark numbers as your own.
    """
    model = YOLO(weights)

    # warm-up (first call includes model/graph setup overhead — exclude it)
    model.predict(sample_image, imgsz=imgsz, verbose=False)

    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(sample_image, imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / n_runs) * 1000
    fps = n_runs / elapsed

    print(f"\nLATENCY (avg over {n_runs} runs, imgsz={imgsz})")
    print(f"  Avg latency: {avg_latency_ms:.2f} ms")
    print(f"  FPS: {fps:.2f}")

    return {"avg_latency_ms": avg_latency_ms, "fps": fps, "imgsz": imgsz, "n_runs": n_runs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--data", required=True, help="Path to dataset YAML (RDD2022-format)")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--latency-sample", default=None, help="Optional: path to one image for latency test")
    parser.add_argument("--out", default="evaluation_results.json")
    args = parser.parse_args()

    results = run_evaluation(args.weights, args.data, args.split, args.imgsz)

    if args.latency_sample:
        results["latency"] = measure_latency(args.weights, args.latency_sample, imgsz=args.imgsz)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved full results to {args.out} — copy the honest numbers (not just the best-looking ones) into your README's 'Results' and 'Limitations' sections.")