"""
ML Concepts – Main Runner
═════════════════════════
Run all or individual concepts:

    python main.py                    # run everything
    python main.py --topic lr         # linear regression only
    python main.py --topic dt         # decision tree only
    python main.py --topic km         # k-means clustering only
    python main.py --topic rf         # bagging & random forest only
    python main.py --topic boost      # boosting only
    python main.py --topic ens        # ensemble methods only
    python main.py --save-plots       # save PNGs to ./plots/
"""

import argparse
import sys
import time
from pathlib import Path

TOPICS = {
    "lr":    ("Linear Regression",      "ml_concepts.models.linear_regression"),
    "dt":    ("Decision Tree",           "ml_concepts.models.decision_tree"),
    "km":    ("K-Means Clustering",      "ml_concepts.models.kmeans_clustering"),
    "rf":    ("Bagging & Random Forest", "ml_concepts.models.bagging_random_forest"),
    "boost": ("Boosting",                "ml_concepts.models.boosting"),
    "ens":   ("Ensemble Methods",        "ml_concepts.models.ensemble"),
}


def run_topic(key, save_plots, plot_dir):
    import importlib
    label, module_path = TOPICS[key]
    print(f"\n{'▶'*3}  {label}  {'◀'*3}")
    t0 = time.perf_counter()
    mod = importlib.import_module(module_path)
    mod.run(save_plots=save_plots, plot_dir=plot_dir)
    elapsed = time.perf_counter() - t0
    print(f"\n  ✓ Completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="ML Concepts Demo")
    parser.add_argument("--topic", choices=list(TOPICS.keys()),
                        help="Run a single topic (default: all)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save plots to ./plots/ instead of displaying them")
    args = parser.parse_args()

    plot_dir = "plots"
    if args.save_plots:
        Path(plot_dir).mkdir(exist_ok=True)
        print(f"Plots will be saved to ./{plot_dir}/")

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  ML CONCEPTS DEMONSTRATION".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    keys = [args.topic] if args.topic else list(TOPICS.keys())
    total_start = time.perf_counter()

    for key in keys:
        run_topic(key, save_plots=args.save_plots, plot_dir=plot_dir)

    total = time.perf_counter() - total_start
    print(f"\n{'═'*60}")
    print(f"  All done in {total:.1f}s")
    if args.save_plots:
        plots = list(Path(plot_dir).glob("*.png"))
        print(f"  {len(plots)} plots saved in ./{plot_dir}/")
        for p in sorted(plots):
            print(f"    • {p.name}")
    print("═" * 60)


if __name__ == "__main__":
    main()
