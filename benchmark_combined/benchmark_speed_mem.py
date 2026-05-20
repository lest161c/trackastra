"""Speed + memory benchmark for all configs. N sweeps up to OOM."""

import csv, logging, sys, time, gc
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
logger=logging.getLogger(__name__)
ROOT=Path(__file__).parent.parent
sys.path.insert(0,str(ROOT/"benchmark_attn")); sys.path.insert(0,str(ROOT/"benchmark_ssl"))
from model_parts import GatherSparseAttention
from track_encoder import AgentCentricNormalization

class SinusoidalPE(nn.Module):
    def __init__(self,d_model,max_len=2048):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        pe[:,0::2]=torch.sin(pos*div);pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

import math
class SparseEnc(nn.Module):
    def __init__(self,d=128,h=4,nl=4,ff=256,dr=0.1,k=16):
        super().__init__();self.k=k;self.pos=SinusoidalPE(d)
        self.ls=nn.ModuleList()
        for _ in range(nl):
            a=GatherSparseAttention(d,h,k,dropout=dr,mode="none")
            l=nn.TransformerEncoderLayer(d,h,ff,dr,activation="gelu",batch_first=True,norm_first=True)
            self.ls.append(nn.ModuleDict({"a":a,"l1":l.linear1,"l2":l.linear2,"n1":l.norm1,"n2":l.norm2,"dp":l.dropout,"d1":l.dropout1,"d2":l.dropout2}))
        self.nm=nn.LayerNorm(d)
    def knn(self,c):
        B,N,_=c.shape;yx=c[...,-2:];d=torch.cdist(yx,yx)
        k=min(self.k,N);_,idx=torch.topk(d,k,dim=-1,largest=False)
        if idx.shape[-1]<self.k:idx=torch.cat([idx,idx[:,:,-1:].expand(-1,-1,self.k-idx.shape[-1])],-1)
        return idx
    def forward(self,x,m=None,c=None):
        x=self.pos(x)
        if c is None:return self.nm(x)
        kn=self.knn(c)
        for l in self.ls:
            a=l["a"](x,x,x,kn,c);x=l["n1"](x+l["d1"](a))
            ff=l["l2"](l["dp"](F.gelu(l["l1"](x))));x=l["n2"](x+l["d2"](ff))
        return self.nm(x)

class DenseEnc(nn.Module):
    def __init__(self,d=128,h=4,nl=4,ff=256,dr=0.1):
        super().__init__();self.pos=SinusoidalPE(d)
        l=nn.TransformerEncoderLayer(d,h,ff,dr,activation="gelu",batch_first=True,norm_first=True)
        self.e=nn.TransformerEncoder(l,nl);self.nm=nn.LayerNorm(d)
    def forward(self,x,m=None,c=None):x=self.pos(x);return self.nm(self.e(x,src_key_padding_mask=m))

class BenchModel(nn.Module):
    def __init__(self,k=None):
        super().__init__();D=128
        self.fp=nn.Linear(7,D);self.cp=nn.Linear(2,D);self.pn=AgentCentricNormalization()
        self.pp=nn.Linear(5,D);self.fu=nn.Sequential(nn.Linear(D*3,D),nn.GELU(),nn.Linear(D,D))
        self.enc=SparseEnc(k=k) if k else DenseEnc()
        self.hd=nn.Linear(D*2,1);self.k=k
    def forward(self,cs,fs,ct,ft,ps=None,pt=None):
        fs_=self.fp(fs);ft_=self.fp(ft);cs_=self.cp(cs);ct_=self.cp(ct)
        p=self.pn(cs,ct);pe=self.pp(p);csx=pe.mean(2);ctx=pe.mean(1)
        s=self.fu(torch.cat([fs_,cs_,csx],-1));t=self.fu(torch.cat([ft_,ct_,ctx],-1))
        b=torch.cat([s,t],1);pd=torch.cat([ps,pt],1)if ps is not None else None
        cb=torch.cat([cs,ct],1)
        e=self.enc(b,m=pd,c=cb if self.k else None)
        N=cs.shape[1];se,te=e[:,:N],e[:,N:]
        B,N1,D=se.shape;N2=te.shape[1]
        return self.hd(torch.cat([se[:,:,None].expand(-1,-1,N2,-1),te[:,None].expand(-1,N1,-1,-1)],-1)).squeeze(-1)

def bench():
    device=torch.device("cuda")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    csv_path=ROOT/"benchmark_combined"/"results"/"speed_mem.csv"
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with open(csv_path,"w",newline="") as f:
        csv.writer(f).writerow(["K","N","time_ms","mem_mb","status"])

    Ns=[128,256,512]
    Ks=[0,4,8,16,32]

    for K in Ks:
        tag="dense" if K==0 else f"K={K}"
        for N in Ns:
            try:
                model=BenchModel(k=K if K>0 else None).to(device)
                B=2
                cs=torch.randn(B,N,2,device=device)*100
                ct=cs+torch.randn(B,N,2,device=device)*3
                fs=torch.randn(B,N,7,device=device)
                ft=torch.randn(B,N,7,device=device)
                a=torch.eye(N,device=device).unsqueeze(0).expand(B,-1,-1).float()
                ps=torch.zeros(B,N,dtype=torch.bool,device=device)
                pt=torch.zeros(B,N,dtype=torch.bool,device=device)
                pw=torch.tensor(10.,device=device)
                opt=torch.optim.AdamW(model.parameters(),lr=3e-4)

                # Warmup
                for _ in range(3):
                    logits=model(cs,fs,ct,ft,ps,pt)
                    loss=F.binary_cross_entropy_with_logits(logits,a,pos_weight=pw)
                    loss.backward();opt.step();opt.zero_grad()

                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                gc.collect();torch.cuda.empty_cache()
                mb=torch.cuda.memory_allocated()

                t0=time.perf_counter()
                for _ in range(10):
                    logits=model(cs,fs,ct,ft,ps,pt)
                    loss=F.binary_cross_entropy_with_logits(logits,a,pos_weight=pw)
                    loss.backward();opt.step();opt.zero_grad()
                torch.cuda.synchronize()
                t=(time.perf_counter()-t0)/10*1000
                mem=(torch.cuda.max_memory_allocated()-mb)/1024**2
                status="ok"
                logger.info(f"  {tag} N={N}: {t:.0f}ms {mem:.0f}MB")
            except RuntimeError as e:
                t,mem=-1,-1;status="oom"
                logger.info(f"  {tag} N={N}: OOM")
            with open(csv_path,"a",newline="") as f:
                csv.writer(f).writerow([K,N,f"{t:.1f}"if t>=0 else"",f"{mem:.1f}"if mem>=0 else"",status])
            del model;gc.collect();torch.cuda.empty_cache()

    logger.info(f"Done: {csv_path}")

if __name__=="__main__":bench()
