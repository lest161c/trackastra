"""Matched downstream: SSL pretrained with same attention as downstream.

Full factorial: K=[0(dense),4,8,16,32] × init=[rand,ssl] × frac=[10%,50%,100%]
SSL checkpoints match downstream attention exactly.
"""
import csv, logging, sys, time, gc, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "benchmark_attn"))
sys.path.insert(0, str(ROOT / "benchmark_ssl"))
from ssl_pipeline import load_experiment_frames, features_from_frame
from track_encoder import AgentCentricNormalization
from model_parts import GatherSparseAttention

# === Model ===
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class SparseEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, n_layers=4, ffn=256, drop=0.1, knn_k=16):
        super().__init__(); self.knn_k = knn_k; self.pos = SinusoidalPE(d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            a = GatherSparseAttention(d_model, nhead, knn_k, dropout=drop, mode="none")
            l = nn.TransformerEncoderLayer(d_model, nhead, ffn, drop, activation="gelu", batch_first=True, norm_first=True)
            self.layers.append(nn.ModuleDict({"attn":a,"linear1":l.linear1,"linear2":l.linear2,
                "norm1":l.norm1,"norm2":l.norm2,"dropout":l.dropout,"dropout1":l.dropout1,"dropout2":l.dropout2}))
        self.norm = nn.LayerNorm(d_model)
    def compute_knn(self, c):
        B,N,_=c.shape; yx=c[...,-2:]; d=torch.cdist(yx,yx)
        k=min(self.knn_k,N); _,idx=torch.topk(d,k=k,dim=-1,largest=False)
        if idx.shape[-1]<self.knn_k: idx=torch.cat([idx,idx[:,:,-1:].expand(-1,-1,self.knn_k-idx.shape[-1])],dim=-1)
        return idx
    def forward(self,x,mask=None,coords=None):
        x=self.pos(x)
        if coords is None: return self.norm(x)
        knn=self.compute_knn(coords)
        for l in self.layers:
            a=l["attn"](x,x,x,knn,coords); x=l["norm1"](x+l["dropout1"](a))
            ff=l["linear2"](l["dropout"](F.gelu(l["linear1"](x)))); x=l["norm2"](x+l["dropout2"](ff))
        return self.norm(x)

class DenseEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, n_layers=4, ffn=256, drop=0.1):
        super().__init__(); self.pos=SinusoidalPE(d_model)
        l=nn.TransformerEncoderLayer(d_model,nhead,ffn,drop,activation="gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(l,n_layers); self.norm=nn.LayerNorm(d_model)
    def forward(self,x,mask=None,coords=None): x=self.pos(x); return self.norm(self.encoder(x,src_key_padding_mask=mask))

class MatchedModel(nn.Module):
    """Uses same attribute names as SSLModel for key match."""
    def __init__(self, k=None):
        super().__init__(); D=128
        self.feat_proj=nn.Linear(7,D); self.coord_proj=nn.Linear(2,D); self.pair_norm=AgentCentricNormalization()
        self.pair_proj=nn.Linear(5,D); self.fusion=nn.Sequential(nn.Linear(D*3,D),nn.GELU(),nn.Linear(D,D))
        sp = SparseEncoder(knn_k=k) if k else DenseEncoder()
        self.encoder = sp
        self.head=nn.Linear(D*2,1); self.sparse_k=k
    def forward(self,cs,fs,ct,ft,ps=None,pt=None):
        fsrc,ftgt=self.feat_proj(fs),self.feat_proj(ft); csrc,ctgt=self.coord_proj(cs),self.coord_proj(ct)
        p=self.pair_norm(cs,ct); pe=self.pair_proj(p); ctx_src,ctx_tgt=pe.mean(2),pe.mean(1)
        s=self.fusion(torch.cat([fsrc,csrc,ctx_src],-1)); t=self.fusion(torch.cat([ftgt,ctgt,ctx_tgt],-1))
        b=torch.cat([s,t],1); pad=torch.cat([ps,pt],1) if ps is not None else None
        cb=torch.cat([cs,ct],1)
        enc=self.encoder(b,mask=pad,coords=cb if self.sparse_k else None)
        Ns=cs.shape[1]; se,te=enc[:,:Ns],enc[:,Ns:]
        B,N1,D=se.shape; N2=te.shape[1]
        return self.head(torch.cat([se[:,:,None].expand(-1,-1,N2,-1),te[:,None].expand(-1,N1,-1,-1)],-1)).squeeze(-1)

# === Data ===
def load_pairs(frames, max_pairs=200):
    pairs=[]
    for i in range(0,len(frames)-1,2):
        try:
            _,_,_,mp,ip=frames[i]; _,_,_,mt,it=frames[i+1]
            from tifffile import imread
            ms,mt=imread(mp),imread(mt)
            def _ld(p): img=imread(p).astype(np.float32); p1,p998=np.percentile(img,(1,99.8)); return np.clip((img-p1)/(p998-p1+1e-8),0,1)
            rs=features_from_frame(ms,_ld(ip)); rt=features_from_frame(mt,_ld(it))
            if rs is None or rt is None: continue
            cs,ls,fd=rs; ct,lt,ftd=rt
            fs=np.concatenate(list(fd.values()),-1).astype(np.float32)
            ft=np.concatenate(list(ftd.values()),-1).astype(np.float32)
            if len(ls)==0 or len(lt)==0: continue
            a=np.zeros((len(ls),len(lt)),np.float32)
            lm={int(l):j for j,l in enumerate(lt)}
            for is_,ls_ in enumerate(ls):
                j=lm.get(int(ls_))
                if j is not None: a[is_,j]=1.0
            pairs.append({"cs":torch.from_numpy(cs).float(),"ct":torch.from_numpy(ct).float(),
                          "fs":torch.from_numpy(fs).float(),"ft":torch.from_numpy(ft).float(),
                          "a":torch.from_numpy(a).float()})
            if len(pairs)>=max_pairs: break
        except: continue
    return pairs

# === Run ===
def run():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    frames=load_experiment_frames(str(ROOT/"data/vanvliet"),conditions=["rpsM","recA","pheA"])
    np.random.seed(42); np.random.shuffle(frames)
    all_pairs=load_pairs(frames)
    val_pairs=all_pairs[:40]; train_pool=all_pairs[40:]
    logger.info(f"Pairs: {len(train_pool)} train, {len(val_pairs)} val")

    csv_path=ROOT/"benchmark_combined"/"results"/"downstream_matched.csv"
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with open(csv_path,"w",newline="") as f:
        csv.writer(f).writerow(["K","init","frac","epoch","train_loss","val_loss","train_acc","val_acc","time_s"])

    K_vals=[0,4,8,16,32]
    fracs=[(1.0,"100%"),(0.5,"50%"),(0.1,"10%")]

    for K in K_vals:
        tag="dense" if K==0 else f"K={K}"
        ssl_path=ROOT/"benchmark_ssl"/"runs"/f"ssl_{tag}"/"best_model.pt"
        ssl_state=None
        if ssl_path.exists():
            ssl_state=torch.load(ssl_path,map_location="cpu",weights_only=False)["model_state_dict"]
        for frac,flabel in fracs:
            n=max(1,int(len(train_pool)*frac)); train_pairs=train_pool[:n]
            for iname,use_ssl in [("rand",False),("ssl",True)]:
                try:
                    model=MatchedModel(k=K if K>0 else None).to(device)
                    if use_ssl and ssl_state is not None:
                        own=model.state_dict(); mapped=0
                        for k in own:
                            if k in ssl_state and own[k].shape==ssl_state[k].shape:
                                own[k]=ssl_state[k].clone(); mapped+=1
                        model.load_state_dict(own,strict=False)
                        logger.info(f"  {tag}+{iname}@{flabel}: {mapped}/{len(own)} keys")
                    params=sum(p.numel() for p in model.parameters())
                    opt=AdamW(model.parameters(),lr=3e-4,weight_decay=0.01); pw=torch.tensor(10.0,device=device)
                    
                    for ep in range(1,11):
                        t0=time.perf_counter(); model.train(); tls,tas=[],[]
                        for b in train_pairs:
                            b={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}
                            opt.zero_grad()
                            logits=model(b["cs"].unsqueeze(0),b["fs"].unsqueeze(0),b["ct"].unsqueeze(0),b["ft"].unsqueeze(0))
                            loss=F.binary_cross_entropy_with_logits(logits,b["a"].unsqueeze(0),pos_weight=pw)
                            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                            tls.append(loss.item())
                            with torch.no_grad():
                                tas.append(((logits.detach()>0).float()==b["a"].unsqueeze(0)).float().mean().item())
                        
                        model.eval(); vls,vas=[],[]
                        with torch.no_grad():
                            for b in val_pairs:
                                b={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}
                                logits=model(b["cs"].unsqueeze(0),b["fs"].unsqueeze(0),b["ct"].unsqueeze(0),b["ft"].unsqueeze(0))
                                loss=F.binary_cross_entropy_with_logits(logits,b["a"].unsqueeze(0),pos_weight=pw)
                                vls.append(loss.item())
                                vas.append(((logits>0).float()==b["a"].unsqueeze(0)).float().mean().item())
                        
                        et=time.perf_counter()-t0
                        tl,ta,vl,va=float(np.mean(tls)),float(np.mean(tas)),float(np.mean(vls)),float(np.mean(vas))
                        with open(csv_path,"a",newline="") as f:
                            csv.writer(f).writerow([K,iname,flabel,ep,f"{tl:.6f}",f"{vl:.6f}",f"{ta:.4f}",f"{va:.4f}",f"{et:.2f}"])
                        if ep in[1,5,10]:
                            logger.info(f"  {tag}+{iname}@{flabel} Ep{ep}: tl={tl:.4f} vl={vl:.4f} [{et:.1f}s]")
                except Exception as e:
                    logger.info(f"  {tag}+{iname}@{flabel}: FAILED {e}")
                finally:
                    del model; gc.collect(); torch.cuda.empty_cache()
    
    logger.info(f"\nDone: {csv_path}")

if __name__=="__main__": run()
