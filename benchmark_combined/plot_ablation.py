"""Comprehensive ablation plots from ablation_full.csv."""

import csv, io, base64
import pandas as pd
import seaborn as sns
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")
figures = []

rows = []
with open("results/ablation_full.csv") as f:
    for r in csv.DictReader(f):
        for k in r:
            try: r[k] = float(r[k]) if r[k] else float("nan")
            except: pass
        rows.append(r)
df = pd.DataFrame(rows)

PAL = {"dense": "#3498db", "sparseK4": "#e67e22", "sparseK16": "#e74c3c"}

# ===== 1. Time scaling (L=1) =====
speed = df[(df["phase"] == "speed") & (df["status"] == "ok")]
fig, ax = plt.subplots(figsize=(9, 5))
for attn in ["dense", "sparseK4", "sparseK16"]:
    sub = speed[(speed["attn"] == attn) & (speed["L"] == 1)]
    sub = sub.sort_values("N")
    ax.plot(sub["N"], sub["time_s"] * 1000, marker="o", label=attn, color=PAL.get(attn, "gray"))
ax.set_xlabel("N (cells)"); ax.set_ylabel("Time per step (ms)")
ax.set_title("Time Scaling: Dense vs Sparse Attention (L=1)")
ax.legend(title="Attention"); ax.grid(True, ls="--", alpha=0.3)
ax.set_xscale("log", base=2); ax.set_yscale("log")
fig.tight_layout(); figures.append(fig); plt.close(fig)

# ===== 2. Time scaling (L=4) =====
fig, ax = plt.subplots(figsize=(9, 5))
for attn in ["dense", "sparseK4", "sparseK16"]:
    sub = speed[(speed["attn"] == attn) & (speed["L"] == 4)]
    sub = sub.sort_values("N")
    ax.plot(sub["N"], sub["time_s"] * 1000, marker="o", label=attn, color=PAL.get(attn, "gray"))
ax.set_xlabel("N (cells)"); ax.set_ylabel("Time per step (ms)")
ax.set_title("Time Scaling: Dense vs Sparse Attention (L=4)")
ax.legend(title="Attention"); ax.grid(True, ls="--", alpha=0.3)
ax.set_xscale("log", base=2); ax.set_yscale("log")
fig.tight_layout(); figures.append(fig); plt.close(fig)

# ===== 3. Memory scaling (L=1) =====
fig, ax = plt.subplots(figsize=(9, 5))
for attn in ["dense", "sparseK4", "sparseK16"]:
    sub = speed[(speed["attn"] == attn) & (speed["L"] == 1)]
    sub = sub.sort_values("N")
    ax.plot(sub["N"], sub["mem_mb"], marker="o", label=attn, color=PAL.get(attn, "gray"))
ax.set_xlabel("N (cells)"); ax.set_ylabel("Incremental GPU Memory (MB)")
ax.set_title("Memory Scaling: Dense vs Sparse (L=1)")
ax.legend(title="Attention"); ax.grid(True, ls="--", alpha=0.3)
ax.set_xscale("log", base=2)
fig.tight_layout(); figures.append(fig); plt.close(fig)

# ===== 4. Convergence L=4 (most realistic) =====
conv = df[(df["phase"] == "converge") & (df["status"] == "ok") & (df["L"] == 4)]
fig, ax = plt.subplots(figsize=(10, 5))
for attn in ["dense", "sparseK4", "sparseK16"]:
    for init in ["rand", "ssl"]:
        sub = conv[(conv["attn"] == attn) & (conv["init"] == init)]
        if sub.empty: continue
        lbl = f"{attn}+{init}"
        ls = "-" if init == "rand" else "--"
        ax.plot(sub["epoch"], sub["val_loss"], marker="o", label=lbl,
                color=PAL.get(attn, "gray"), linestyle=ls, markersize=4)
ax.set_title("Convergence at N=512, L=4: All Variants")
ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss")
ax.legend(title="Variant", fontsize=7); ax.grid(True, ls="--", alpha=0.3)
fig.tight_layout(); figures.append(fig); plt.close(fig)

# ===== 5. Convergence L=1 =====
conv1 = df[(df["phase"] == "converge") & (df["status"] == "ok") & (df["L"] == 1)]
fig, ax = plt.subplots(figsize=(10, 5))
for attn in ["dense", "sparseK4", "sparseK16"]:
    for init in ["rand", "ssl"]:
        sub = conv1[(conv1["attn"] == attn) & (conv1["init"] == init)]
        if sub.empty: continue
        lbl = f"{attn}+{init}"
        ls = "-" if init == "rand" else "--"
        ax.plot(sub["epoch"], sub["val_loss"], marker="o", label=lbl,
                color=PAL.get(attn, "gray"), linestyle=ls, markersize=4)
ax.set_title("Convergence at N=512, L=1: All Variants")
ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss")
ax.legend(title="Variant", fontsize=7); ax.grid(True, ls="--", alpha=0.3)
fig.tight_layout(); figures.append(fig); plt.close(fig)

# ===== 6. Final loss bar (L=4, epoch 15) =====
final = conv[(conv["epoch"] == 15) & (conv["L"] == 4)]
fig, ax = plt.subplots(figsize=(9, 4))
labels = final["attn"] + "+" + final["init"]
colors = [PAL.get(a, "gray") for a in final["attn"]]
bars = ax.bar(range(len(final)), final["val_loss"], color=colors, width=0.5)
ax.set_xticks(range(len(final))); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Val Loss (epoch 15)"); ax.set_title("Final Val Loss at N=512, L=4")
for i, (_, r) in enumerate(final.iterrows()):
    ax.text(i, r["val_loss"] + 0.01, f"{r['val_loss']:.3f}", ha="center", fontsize=8)
ax.grid(True, ls="--", alpha=0.3, axis="y")
fig.tight_layout(); figures.append(fig); plt.close(fig)

# ===== 7. Epoch 1 bar =====
ep1 = conv[conv["epoch"] == 1]
fig, ax = plt.subplots(figsize=(9, 4))
labels = ep1["attn"] + "+" + ep1["init"]
colors = [PAL.get(a, "gray") for a in ep1["attn"]]
ax.bar(range(len(ep1)), ep1["val_loss"], color=colors, width=0.5)
ax.set_xticks(range(len(ep1))); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Val Loss (epoch 1)"); ax.set_title("Epoch 1 Val Loss at N=512, L=4")
for i, (_, r) in enumerate(ep1.iterrows()):
    ax.text(i, r["val_loss"] + 0.02, f"{r['val_loss']:.3f}", ha="center", fontsize=8)
ax.grid(True, ls="--", alpha=0.3, axis="y")
fig.tight_layout(); figures.append(fig); plt.close(fig)

# ===== 8. Speedup table =====
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
rows_t = [["Metric", "Dense", "Sparse K=4", "Sparse K=16", "Best"]]
for L in [1, 4]:
    for N in [128, 256, 512]:
        d = speed[(speed["attn"]=="dense") & (speed["L"]==L) & (speed["N"]==N)]
        s4 = speed[(speed["attn"]=="sparseK4") & (speed["L"]==L) & (speed["N"]==N)]
        s16 = speed[(speed["attn"]=="sparseK16") & (speed["L"]==L) & (speed["N"]==N)]
        if d.empty: continue
        dt = d["time_s"].values[0] * 1000
        s4t = s4["time_s"].values[0] * 1000 if not s4.empty else float("nan")
        s16t = s16["time_s"].values[0] * 1000 if not s16.empty else float("nan")
        best = min([v for v in [s4t, s16t] if not np.isnan(v)], default=dt)
        rows_t.append([f"L={L} N={N} (ms)", f"{dt:.0f}", f"{s4t:.0f}", f"{s16t:.0f}", f"{best:.0f}"])
table = ax.table(cellText=rows_t, cellLoc="center", loc="center", colWidths=[0.25]*5)
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.5)
ax.set_title("Speed Comparison: ms/step (lower = better)", fontsize=12, pad=15)
fig.tight_layout(); figures.append(fig); plt.close(fig)

# ===== 9. Convergence speedup table =====
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
rows_c = [["Variant", "Epoch 1 Loss", "Epoch 15 Loss", "Final / Init Ratio"]]
for attn in ["dense", "sparseK4", "sparseK16"]:
    for init in ["rand", "ssl"]:
        sub = conv[(conv["attn"]==attn) & (conv["init"]==init)]
        if sub.empty: continue
        e1 = sub[sub["epoch"]==1]["val_loss"].values[0]
        e15 = sub[sub["epoch"]==15]["val_loss"].values[0]
        ratio = e15 / e1 if e1 > 0 else 0
        rows_c.append([f"{attn}+{init}", f"{e1:.3f}", f"{e15:.3f}", f"{ratio:.2f}"])
table = ax.table(cellText=rows_c, cellLoc="center", loc="center", colWidths=[0.25]*4)
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.5)
ax.set_title("Convergence at N=512, L=4 (lower = better)", fontsize=12, pad=15)
fig.tight_layout(); figures.append(fig); plt.close(fig)

# HTML
def fig_to_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0); return base64.b64encode(buf.read()).decode()

html = [
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<title>Full Ablation: Attention × K × L × Init</title>",
    "<style>body{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#fafafa}</style>",
    "</head><body>",
    "<h1>Full Ablation Study</h1>",
    "<p>All combinations: Dense / SparseK4 / SparseK16 × L=1/4 × N=128-1024 × Random/SSL init.</p>",
    "<p>GPU: RTX A500 Laptop (4GB). All variants OOM at N=1024.</p>",
]
for i, fig in enumerate(figures):
    b64 = fig_to_b64(fig)
    html.append(f"<figure><figcaption>Figure {i+1}</figcaption><img src='data:image/png;base64,{b64}' /></figure>")
html.append("</body></html>")

with open("ablation_full.html", "w") as f:
    f.write("\n".join(html))
print(f"Saved ablation_full.html with {len(figures)} figures")
