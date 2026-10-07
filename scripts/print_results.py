from __future__ import annotations
import json
from pathlib import Path

p=Path("results/summaries/results_summary_v1.1.json")
d=json.loads(p.read_text(encoding="utf-8"))
i=d["internal"]["enhanced"]; e=d["external"]["enhanced"]; o=d["ood"]
print(f"Accuracy: {i['accuracy']:.2%}")
print(f"Macro-F1: {i['macro_f1']:.2%}")
print(f"Macro-AUROC: {i['macro_auroc']:.2%}")
print(f"Mpox recall: {d['internal']['mpox_recall_enhanced']:.2%}")
print(f"External restricted-4 accuracy: {e['restricted_four_class_accuracy']:.2%}")
print(f"Held-out OOD AUROC: {o['test_auroc_mean']:.3f} ± {o['test_auroc_std']:.3f}")
