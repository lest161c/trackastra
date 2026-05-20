"""Combined: sparse+SSL vs dense+rand at N=8192 on 4GB.

Tests at B=1 to fit pair_norm O(N²) within 4GB budget.
Times + memory + convergence for all variants at N=8192.
"""

import csv, logging, sys, time, gc, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
logger=logging.getLogger(__name__)

ROOT=Path(__file__).parent.parent
sys.path.insert(0,str(ROOT/"benchmark_attn")); sys.path.insert(0,str(ROOT/"benchmark_ssl"))
from model_parts import GatherSparseAttention
from track_encoder import AgentCentricNormalization

class SinPE(nn.Module):
    def __init__(self,d,L=2048):
        super().__init__();pe=torch.zeros(L,d)
        p=torch.arange(0,L,dtype=torch.float).unsqueeze(1)
        t=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.)/d))
        pe[:,0::2]=torch.sin(p*t);pe[:,1::2]=torch.cos(p*t);self.register_buffer("p",pe.unsqueeze(0))
    def forward(self,x):return x+self.p[:,:x.size(1)]

class FullModel(nn.Module):
    def __init__(self,k=None,D=128,H=4,L=4):
        super().__init__()
        self.fp=nn.Linear(7,D);self.cp=nn.Linear(2,D);self.pn=AgentCentricNormalization()
        self.pp=nn.Linear(5,D);self.fu=nn.Sequential(nn.Linear(D*3,D),nn.GELU(),nn.Linear(D,D))
        self.k=k
        if k:
            class SpEnc(nn.Module):
                def __init__(s):super().__init__();s.pe=SinPE(D);s.ls=nn.ModuleList()
                for _ in range(L):
                    a=GatherSparseAttention(D,H,k,dropout=0.,mode="none")
                    l=nn.TransformerEncoderLayer(D,H,256,0.,activation="gelu",batch_first=True,norm_first=True)
                    s.ls.append(nn.ModuleDict({"a":a,"l1":l.linear1,"l2":l.linear2,"n1":l.norm1,"n2":l.norm2,"dp":l.dropout,"d1":l.dropout1,"d2":l.dropout2}))
                def knn(s,c):
                    B,N,_=c.shape;yx=c[...,-2:];d=torch.cdist(yx,yx)
                    kk=min(k,N);_,idx=torch.topk(d,kk,dim=-1,largest=False)
                    if idx.shape[-1]<k:idx=torch.cat([idx,idx[:,:,-1:].expand(-1,-1,33-idx.shape[-1])],-1)
                    return idx
                def forward(s,x,m=None,c=None):
                    x=s.pe(x)
                    if c is None:return x
                    kn=s.knn(c)
                    for lyr in s.ls:
                        a=lyr["a"](x,x,x,kn,c);x=lyr["n1"](x+lyr["d1"](a))
                        ff=lyr["l2"](lyr["dp"](F.gelu(lyr["l1"](x))));x=lyr["n2"](x+lyr["d2"](ff))
                    return x
            self.enc=SpEnc()
        else:
            class DsEnc(nn.Module):
                def __init__(s):super().__init__();s.pe=SinPE(D)
                l=nn.TransformerEncoderLayer(D,H,256,0.,activation="gelu",batch_first=True,norm_first=True)
                s.e=nn.TransformerEncoder(l,L)
                def forward(s,x,m=None,c=None):x=s.pe(x);return s.e(x,src_key_padding_mask=m)
            self.enc=DsEnc()
        self.hd=nn.Linear(D*2,1)

    def forward(self,cs,fs,ct,ft):
        fsrc,ftgt=self.fp(fs),self.fp(ft);csrc,ctgt=self.cp(cs),self.cp(ct)
        p=self.pn(cs,ct);pe=self.pp(p);csx,ctx=pe.mean(2),pe.mean(1)
        s=self.fu(torch.cat([fsrc,csrc,csx],-1));t=self.fu(torch.cat([ftgt,ctgt,ctx],-1))
        b=torch.cat([s,t],1);cb=torch.cat([cs,ct],1)
        e=self.enc(b,c=cb if self.k else None)
        N=cs.shape[1];se,te=e[:,:N],e[:,N:]
        B,N1,D=se.shape;N2=te.shape[1]
        return self.hd(torch.cat([se[:,:,None].expand(-1,-1,N2,-1),te[:,None].expand(-1,N1,-1,-1)],-1)).squeeze(-1)

SSL_PATH=ROOT/"benchmark_ssl"/"runs"/"ssl_K=16"/"best_model.pt"

def init_ssl(m):
    sd=m.state_dict();ss=torch.load(SSL_PATH,map_location="cpu",weights_only=False)["model_state_dict"];c=0
    for k in sd:
        if k in ss and sd[k].shape==ss[k].shape:sd[k]=ss[k].clone();c+=1
    m.load_state_dict(sd,strict=False);logger.info(f"SSL: {c}/{len(sd)} keys")

def run():
    device=torch.device("cuda");logger.info(f"{torch.cuda.get_device_name(0)}")

    N=8192;E=5;B=1
    csv_path=ROOT/"benchmark_combined"/"results"/"combined_n8192.csv"
    csv_path.parent.mkdir(parents=True,exist_ok=True)

    with open(csv_path,"w",newline="") as f:
        csv.writer(f).writerow(["variant","epoch","train_loss","val_loss","time_s","mem_mb","status"])

    # Generate data once
    cs=torch.randn(B,N,2,device=device)*100
    ct=cs+torch.randn(B,N,2,device=device)*3
    fs=torch.randn(B,N,7,device=device);ft=torch.randn(B,N,7,device=device)
    a=torch.eye(N,device=device).unsqueeze(0).float()
    pw=torch.tensor(10.,device=device)

    for variant,k,use_ssl in [("dense+rand",None,False),("sparseK16+rand",16,False),("sparseK16+ssl",16,True)]:
        try:
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            mb=torch.cuda.memory_allocated()
            m=FullModel(k=k).to(device)
            if use_ssl:init_ssl(m)
            opt=AdamW(m.parameters(),lr=3e-4)

            # Warmup + measure peak mem
            logits=m(cs,fs,ct,ft)
            loss=F.binary_cross_entropy_with_logits(logits,a,pos_weight=pw)
            loss.backward();opt.step();opt.zero_grad()
            mem=(torch.cuda.max_memory_allocated()-mb)/1024**2

            for ep in range(1,E+1):
                t0=time.perf_counter()
                for _ in range(3):
                    logits=m(cs,fs,ct,ft)
                    loss=F.binary_cross_entropy_with_logits(logits,a,pos_weight=pw)
                    loss.backward();opt.step();opt.zero_grad()
                torch.cuda.synchronize()
                et=(time.perf_counter()-t0)/3
                tl=loss.item()
                with torch.no_grad():
                    logits=m(cs,fs,ct,ft)
                    vl=F.binary_cross_entropy_with_logits(logits,a,pos_weight=pw).item()
                logger.info(f"  {variant} Ep{ep}: tl={tl:.4f} vl={vl:.4f} {et*1000:.0f}ms/step {mem:.0f}MB")
                with open(csv_path,"a",newline="") as f:
                    csv.writer(f).writerow([variant,ep,f"{tl:.6f}",f"{vl:.6f}",f"{et:.6f}",f"{mem:.1f}","ok"])
        except RuntimeError as e:
            logger.info(f"  {variant}: OOM/ERR - {e}")
            with open(csv_path,"a",newline="") as f:
                csv.writer(f).writerow([variant,0,"","","","","oom"])
        del m;gc.collect();torch.cuda.empty_cache()

    logger.info(f"Done: {csv_path}")

if __name__=="__main__":run()
