"""Plots for 2x2 + K ablation benchmark."""

import csv, io, base64
import pandas as pd
import seaborn as sns
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")
figures = []

rows = []
with open("results/combined_benchmark.csv") as f:
    for r in csv.DictReader(f):
        for k in r:
            try: r[k] = float(r[k])
            except ValueError: pass
        rows.append(r)
df = pd.DataFrame(rows)

PAL = {"dense+rand": "#3498db", "dense+ssl": "#2ecc71",
        "sparseK4+rand": "#e67e22", "sparseK4+ssl": "#f39c12",
        "sparseK16+rand": "#e74c3c", "sparseK16+ssl": "#9b59b6"}
ORDER = ["dense+rand", "dense+ssl", "sparseK4+rand", "sparseK4+ssl", "sparseK16+rand", "sparseK16+ssl"]

# 1. Val Loss
fig, ax = plt.subplots(figsize=(10, 5))
for v in ORDER:
    sub = df[df["variant"] == v]
    sns.lineplot(data=sub, x="epoch", y="val_loss", marker="o", label=v, color=PAL[v], ax=ax)
ax.set_title("Combined: Validation Loss (K=4 vs K=16)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss")
ax.legend(title="Variant", fontsize=8); ax.grid(True, ls="--", alpha=0.3)
fig.tight_layout(); figures.append(fig); plt.close(fig)

# 2. Epoch 1 val loss bar
ep1 = df[df["epoch"] == 1]
fig, ax = plt.subplots(figsize=(10, 4))
colors = [PAL[v] for v in ep1["variant"]]
ax.bar(range(len(ep1)), ep1["val_loss"], color=colors, width=0.5)
ax.set_xticks(range(len(ep1))); ax.set_xticklabels(ep1["variant"], rotation=15)
ax.set_ylabel("Epoch 1 Val Loss"); ax.set_title("Epoch 1 Initial Val Loss (lower = better)")
for i, (_, r) in enumerate(ep1.iterrows()):
    ax.text(i, r["val_loss"] + 0.0005, f"{r['val_loss']:.4f}", ha="center", fontsize=9)
ax.grid(True, ls="--", alpha=0.3, axis="y")
fig.tight_layout(); figures.append(fig); plt.close(fig)

# 3. Best val loss bar
best = df.loc[df.groupby("variant")["val_loss"].idxmin()]
fig, ax = plt.subplots(figsize=(10, 4))
colors = [PAL[v] for v in best["variant"]]
ax.bar(range(len(best)), best["val_loss"], color=colors, width=0.5)
ax.set_xticks(range(len(best))); ax.set_xticklabels(best["variant"], rotation=15)
ax.set_ylabel("Best Val Loss"); ax.set_title("Best Val Loss Achieved (lower = better)")
for i, (_, r) in enumerate(best.iterrows()):
    ax.text(i, r["val_loss"] + 0.0002, f"{r['val_loss']:.4f}", ha="center", fontsize=9)
ax.grid(True, ls="--", alpha=0.3, axis="y")
fig.tight_layout(); figures.append(fig); plt.close(fig)

# 4. Time per epoch
fig, ax = plt.subplots(figsize=(10, 4))
for v in ORDER:
    sub = df[df["variant"] == v]
    ax.plot(sub["epoch"], sub["time_s"], marker="o", label=v, color=PAL[v])
ax.set_title("Time per Epoch")
ax.set_xlabel("Epoch"); ax.set_ylabel("Time (s)")
ax.legend(title="Variant", fontsize=8); ax.grid(True, ls="--", alpha=0.3)
fig.tight_layout(); figures.append(fig); plt.close(fig)

# 5. Cumulative time to target
target = 0.030
fig, ax = plt.subplots(figsize=(10, 5))
for v in ORDER:
    sub = df[df["variant"] == v].copy()
    sub["cum"] = sub["time_s"].cumsum()
    ax.plot(sub["epoch"], sub["cum"], marker="o", label=v, color=PAL[v])
    hit = sub[sub["val_loss"] <= target]
    if len(hit) > 0:
        h = hit.iloc[0]
        ax.scatter([h["epoch"]], [h["cum"]], color=PAL[v], s=80, zorder=5)
        ax.annotate(f"{h['cum']:.1f}s", (h["epoch"], h["cum"]),
                     textcoords="offset points", xytext=(8, 8), fontsize=7, color=PAL[v])
ax.set_title(f"Wall Time to Val Loss ≤ {target}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Cumulative Time (s)")
ax.legend(title="Variant", fontsize=8); ax.grid(True, ls="--", alpha=0.3)
fig.tight_layout(); figures.append(fig); plt.close(fig)

# 6. K comparison: K=4 vs K=16 (random only, epoch 1)
k_comp = df[df["epoch"] == 1]
k_comp = k_comp[k_comp["variant"].str.contains("rand")]
fig, ax = plt.subplots(figsize=(7, 4))
x = range(3); labels = ["Dense", "Sparse K=4", "Sparse K=16"]
vals = [k_comp[k_comp["variant"] == v]["val_loss"].values[0] for v in ["dense+rand", "sparseK4+rand", "sparseK16+rand"]]
ax.bar(x, vals, color=["#3498db", "#e67e22", "#e74c3c"], width=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Epoch 1 Val Loss"); ax.set_title("Effect of K on Initial Convergence (Random Init)")
for i, v in enumerate(vals):
    ax.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=10)
ax.grid(True, ls="--", alpha=0.3, axis="y")
fig.tight_layout(); figures.append(fig); plt.close(fig)

# HTML
def fig_to_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0); return base64.b64encode(buf.read()).decode()

html = [
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<title>Combined Benchmark: Sparse + SSL + K ablation</title>",
    "<style>body{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#fafafa}</style>",
    "</head><body>",
    "<h1>Combined Benchmark: Sparse Attention + SSL + K Ablation</h1>",
    "<p>6 variants: Dense, Sparse K=4, Sparse K=16 × Random/SSL init.</p>",
]
for i, fig in enumerate(figures):
    b64 = fig_to_b64(fig)
    html.append(f"<figure><figcaption>Figure {i+1}</figcaption><img src='data:image/png;base64,{b64}' /></figure>")
html.append("</body></html>")

with open("benchmark_combined.html", "w") as f:
    f.write("\n".join(html))
print(f"Saved benchmark_combined.html with {len(figures)} figures")
