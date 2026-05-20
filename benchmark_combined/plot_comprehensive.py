"""Comprehensive: convergence + speed + memory. All K × init × N."""

import csv,io,base64
import pandas as pd
import seaborn as sns
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid"); figures=[]

# === Load data ===
conv_rows=[]
with open("results/downstream_matched.csv") as f:
    for r in csv.DictReader(f):
        for k in r:
            try:r[k]=float(r[k])if r[k]else float("nan")
            except:pass
        conv_rows.append(r)
conv=pd.DataFrame(conv_rows)

spd_rows=[]
with open("results/speed_mem.csv") as f:
    for r in csv.DictReader(f):
        for k in r:
            try:r[k]=float(r[k])if r[k]else float("nan")
            except:pass
        spd_rows.append(r)
spd=pd.DataFrame(spd_rows)

PAL={"dense":"#3498db","K=4":"#e67e22","K=8":"#2ecc71","K=16":"#e74c3c","K=32":"#9b59b6"}

# --- 1. Time vs N ---
fig,ax=plt.subplots(figsize=(9,5))
for k in [0,4,8,16,32]:
    sub=spd[spd["K"]==k].sort_values("N")
    lbl="dense" if k==0 else f"K={k}"
    ax.plot(sub["N"],sub["time_ms"],marker="o",label=lbl,color=PAL[lbl])
ax.set_xlabel("N (cells)");ax.set_ylabel("Time per step (ms)")
ax.set_title("Speed: Time vs N (all K)");ax.legend(title="Config")
ax.set_xscale("log",base=2);ax.set_yscale("log");ax.grid(True,ls="--",alpha=0.3)
fig.tight_layout();figures.append(fig);plt.close(fig)

# --- 2. Memory vs N ---
fig,ax=plt.subplots(figsize=(9,5))
for k in [0,4,8,16,32]:
    sub=spd[spd["K"]==k].sort_values("N")
    lbl="dense" if k==0 else f"K={k}"
    ax.plot(sub["N"],sub["mem_mb"],marker="o",label=lbl,color=PAL[lbl])
ax.set_xlabel("N (cells)");ax.set_ylabel("Peak GPU Memory (MB)")
ax.set_title("Memory: GPU Memory vs N (all K)");ax.legend(title="Config")
ax.set_xscale("log",base=2);ax.grid(True,ls="--",alpha=0.3)
fig.tight_layout();figures.append(fig);plt.close(fig)

# --- 3. Time vs K bar (N=512) ---
sub=spd[(spd["N"]==512)&(spd["status"]=="ok")]
fig,ax=plt.subplots(figsize=(8,4))
labels=[("dense" if r["K"]==0 else f"K={int(r['K'])}") for _,r in sub.iterrows()]
colors=[PAL[l] for l in labels]
ax.bar(range(len(sub)),sub["time_ms"],color=colors,width=0.5)
ax.set_xticks(range(len(sub)));ax.set_xticklabels(labels)
ax.set_ylabel("ms/step");ax.set_title("Time per Step at N=512")
for i,(_,r) in enumerate(sub.iterrows()):
    ax.text(i,r["time_ms"]+1,f"{r['time_ms']:.0f}ms",ha="center",fontsize=9)
ax.grid(True,ls="--",alpha=0.3,axis="y")
fig.tight_layout();figures.append(fig);plt.close(fig)

# --- 4. Memory vs K bar (N=512) ---
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(range(len(sub)),sub["mem_mb"],color=colors,width=0.5)
ax.set_xticks(range(len(sub)));ax.set_xticklabels(labels)
ax.set_ylabel("Peak GPU Memory (MB)");ax.set_title("GPU Memory at N=512")
for i,(_,r) in enumerate(sub.iterrows()):
    ax.text(i,r["mem_mb"]+10,f"{r['mem_mb']:.0f}MB",ha="center",fontsize=9)
ax.grid(True,ls="--",alpha=0.3,axis="y")
fig.tight_layout();figures.append(fig);plt.close(fig)

# --- 5. Convergence at 10% data (SSL focus) ---
conv10=conv[(conv["frac"]=="10%")&(conv["epoch"]<=10)]
fig,ax=plt.subplots(figsize=(10,5))
for k_init in set(tuple(x) for x in conv10[["K","init"]].drop_duplicates().values):
    k,init_=int(k_init[0]),k_init[1]
    sub=conv10[(conv10["K"]==k)&(conv10["init"]==init_)];lbl=f"{'dense'if k==0 else f'K={k}'}+{init_}"
    ls="-" if init_=="rand" else "--"
    ax.plot(sub["epoch"],sub["val_loss"],marker="o",label=lbl,color=PAL.get(lbl.split("+")[0],"gray"),linestyle=ls,markersize=4)
ax.set_xlabel("Epoch");ax.set_ylabel("Val Loss");ax.set_title("Convergence at 10% Data (low-label regime)")
ax.legend(title="Config",fontsize=7);ax.grid(True,ls="--",alpha=0.3)
fig.tight_layout();figures.append(fig);plt.close(fig)

# --- 6. Convergence at 50% data ---
conv50=conv[(conv["frac"]=="50%")&(conv["epoch"]<=10)]
fig,ax=plt.subplots(figsize=(10,5))
for k_init in set(tuple(x) for x in conv50[["K","init"]].drop_duplicates().values):
    k,init_=int(k_init[0]),k_init[1]
    sub=conv50[(conv50["K"]==k)&(conv50["init"]==init_)];lbl=f"{'dense'if k==0 else f'K={k}'}+{init_}"
    ls="-" if init_=="rand" else "--"
    ax.plot(sub["epoch"],sub["val_loss"],marker="o",label=lbl,color=PAL.get(lbl.split("+")[0],"gray"),linestyle=ls,markersize=4)
ax.set_xlabel("Epoch");ax.set_ylabel("Val Loss");ax.set_title("Convergence at 50% Data")
ax.legend(title="Config",fontsize=7);ax.grid(True,ls="--",alpha=0.3)
fig.tight_layout();figures.append(fig);plt.close(fig)

# --- 7. Convergence at 100% data ---
conv100=conv[(conv["frac"]=="100%")&(conv["epoch"]<=10)]
fig,ax=plt.subplots(figsize=(10,5))
for k_init in set(tuple(x) for x in conv100[["K","init"]].drop_duplicates().values):
    k,init_=int(k_init[0]),k_init[1]
    sub=conv100[(conv100["K"]==k)&(conv100["init"]==init_)];lbl=f"{'dense'if k==0 else f'K={k}'}+{init_}"
    ls="-" if init_=="rand" else "--"
    ax.plot(sub["epoch"],sub["val_loss"],marker="o",label=lbl,color=PAL.get(lbl.split("+")[0],"gray"),linestyle=ls,markersize=4)
ax.set_xlabel("Epoch");ax.set_ylabel("Val Loss");ax.set_title("Convergence at 100% Data")
ax.legend(title="Config",fontsize=7);ax.grid(True,ls="--",alpha=0.3)
fig.tight_layout();figures.append(fig);plt.close(fig)

# --- 8. SSL benefit heatmap (10% data, epoch 10) ---
ep10=conv[(conv["epoch"]==10)&(conv["frac"]=="10%")]
ssl_benefit={}
for k in [0,4,8,16,32]:
    r=ep10[(ep10["K"]==k)&(ep10["init"]=="rand")]["val_loss"].values
    s=ep10[(ep10["K"]==k)&(ep10["init"]=="ssl")]["val_loss"].values
    if len(r) and len(s):
        ssl_benefit[k]=float(r[0]-s[0])
fig,ax=plt.subplots(figsize=(7,3))
ks=list(ssl_benefit.keys());vals=[ssl_benefit[k]for k in ks]
labels2=[("dense" if k==0 else f"K={k}")for k in ks]
colors2=["#2ecc71"if v>0 else"#e74c3c"for v in vals]
ax.bar(range(len(ks)),vals,color=colors2,width=0.5)
ax.axhline(y=0,color="gray",ls="--")
ax.set_xticks(range(len(ks)));ax.set_xticklabels(labels2)
ax.set_ylabel("Δ Val Loss (rand - ssl)");ax.set_title("SSL Benefit at 10% Data (positive = SSL helps)")
for i,v in enumerate(vals):
    ax.text(i,v+0.02 if v>0 else v-0.06,f"{v:+.3f}",ha="center",fontsize=9)
ax.grid(True,ls="--",alpha=0.3,axis="y")
fig.tight_layout();figures.append(fig);plt.close(fig)

# --- 9. Speed/convergence tradeoff (at N=512, 10% data) ---
final_10=conv[(conv["epoch"]==10)&(conv["frac"]=="10%")&(conv["init"]=="rand")]
fig,ax=plt.subplots(figsize=(8,5))
for _,r in final_10.iterrows():
    k=int(r["K"]);lbl="dense" if k==0 else f"K={k}"
    t=spd[(spd["K"]==k)&(spd["N"]==512)&(spd["status"]=="ok")]["time_ms"].values
    if len(t)==0:continue
    ax.scatter(t[0],r["val_loss"],s=150,color=PAL[lbl],label=lbl,zorder=5)
    ax.annotate(lbl,(t[0],r["val_loss"]),textcoords="offset points",xytext=(10,-5),fontsize=9)
ax.set_xlabel("Time per Step (ms)");ax.set_ylabel("Val Loss (epoch 10, 10% data)")
ax.set_title("Speed-Convergence Tradeoff at N=512")
ax.grid(True,ls="--",alpha=0.3)
fig.tight_layout();figures.append(fig);plt.close(fig)

# --- 10. Summary table ---
fig,ax=plt.subplots(figsize=(12,4));ax.axis("off")
rows=[["Config","Time(N=512)","Mem(N=512)","ValLoss(10%data,ep10)","ValLoss(100%data,ep10)","SSL Benefit"]]
for k in [0,4,8,16,32]:
    lbl="dense" if k==0 else f"K={k}"
    t=spd[(spd["K"]==k)&(spd["N"]==512)&(spd["status"]=="ok")]["time_ms"].values
    m=spd[(spd["K"]==k)&(spd["N"]==512)&(spd["status"]=="ok")]["mem_mb"].values
    v10r=conv[(conv["K"]==k)&(conv["epoch"]==10)&(conv["frac"]=="10%")&(conv["init"]=="rand")]["val_loss"].values
    v10s=conv[(conv["K"]==k)&(conv["epoch"]==10)&(conv["frac"]=="10%")&(conv["init"]=="ssl")]["val_loss"].values
    v100=conv[(conv["K"]==k)&(conv["epoch"]==10)&(conv["frac"]=="100%")&(conv["init"]=="rand")]["val_loss"].values
    tm=f"{t[0]:.0f}ms"if len(t)else"-";mm=f"{m[0]:.0f}MB"if len(m)else"-"
    v10r_s=f"{v10r[0]:.3f}"if len(v10r)else"-"
    v100_s=f"{v100[0]:.3f}"if len(v100)else"-"
    ssl_b=f"{(v10r[0]-v10s[0]):+.3f}"if len(v10r)and len(v10s)else"-"
    rows.append([lbl,tm,mm,v10r_s,v100_s,ssl_b])
tbl=ax.table(cellText=rows,cellLoc="center",loc="center",colWidths=[0.15]*6)
tbl.auto_set_font_size(False);tbl.set_fontsize(9);tbl.scale(1,1.6)
ax.set_title("Full Factorial Summary: All Metrics",fontsize=13,pad=15)
fig.tight_layout();figures.append(fig);plt.close(fig)

# --- HTML ---
def b64(fig):
    b=io.BytesIO();fig.savefig(b,format="png",dpi=130,bbox_inches="tight");b.seek(0);return base64.b64encode(b.read()).decode()

html=[
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<title>Comprehensive Benchmark: All K × Init × N × Data</title>",
    "<style>body{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#fafafa}</style>",
    "</head><body>",
    "<h1>Comprehensive Benchmark: Full Factorial</h1>",
    "<p>K=0(dense),4,8,16,32 × Random/SSL init × 10/50/100% data × N=128-512.</p>",
]
for i,fig in enumerate(figures):
    html.append(f"<figure><figcaption>Figure {i+1}</figcaption><img src='data:image/png;base64,{b64(fig)}' /></figure>")
html.append("</body></html>")

with open("benchmark_comprehensive.html","w") as f:
    f.write("\n".join(html))
print(f"Saved benchmark_comprehensive.html with {len(figures)} figures")
