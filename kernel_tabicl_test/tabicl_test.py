import subprocess, sys, glob, time, numpy as np
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
# offline tabicl (from dataset) + online tabpfn (internet ON)
wh=glob.glob("/kaggle/input/**/tabicl-*.whl", recursive=True)+glob.glob("/kaggle/input/**/einops-*.whl", recursive=True)
subprocess.run([sys.executable,"-m","pip","install","-q","--no-index","--no-deps",*wh], check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","--no-deps","tabpfn"], check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","huggingface_hub","einops"], check=False)
ck=glob.glob("/kaggle/input/**/tabicl-regressor-*.ckpt", recursive=True)[0]
d=np.load(glob.glob("/kaggle/input/**/rogii_tabicl_data.npz", recursive=True)[0])
X,y,g=d["X"],d["y"],d["g"]; truth,anchor,fusion=d["truth"],d["anchor"],d["fusion"]
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
def rmse(t,p): return float(np.sqrt(np.mean((t-p)**2)))
from tabicl import TabICLRegressor; log("tabicl ok")
TABPFN=None
try:
    from tabpfn import TabPFNRegressor; TABPFN=TabPFNRegressor; log("tabpfn imported")
except Exception as e:
    log(f"tabpfn IMPORT FAIL: {repr(e)[:160]}")
rng=np.random.default_rng(0); ot=np.full(len(y),np.nan); op=np.full(len(y),np.nan); og=np.full(len(y),np.nan)
for fi,(tr,va) in enumerate(GroupKFold(5).split(X,groups=g)):
    if fi>=3: break
    ci=rng.choice(tr,min(6000,len(tr)),replace=False); vi=rng.choice(va,min(5000,len(va)),replace=False)
    try:
        r=TabICLRegressor(model_path=ck,allow_auto_download=False,device="cpu",n_estimators=4,batch_size=4); r.fit(X[ci],y[ci]); ot[vi]=r.predict(X[vi])
    except Exception as e: log(f"tabicl f{fi} err {repr(e)[:120]}")
    if TABPFN is not None:
        try:
            cp=rng.choice(tr,min(4000,len(tr)),replace=False)
            rp=TABPFN(device="cpu"); rp.fit(X[cp],y[cp]); op[vi]=rp.predict(X[vi]); log(f"tabpfn f{fi} ok")
        except Exception as e: log(f"tabpfn f{fi} ERR {repr(e)[:160]}")
    og[vi]=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=63,min_child_samples=200,reg_lambda=1.0,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,n_jobs=-1,verbose=-1).fit(X[tr[:60000]],y[tr[:60000]]).predict(X[vi])
m=np.isfinite(ot)&np.isfinite(og)&np.isfinite(op)
def sc(p): return rmse(truth[m], 0.75*(anchor[m]+p[m])+0.25*fusion[m])
log(f"eval rows {int(m.sum())}")
if m.sum()>0:
    print(f"  GBM {sc(og):.4f}  TabICL {sc(ot):.4f}  TabPFN {sc(op):.4f}", flush=True)
    print(f"  corr(TabICL,TabPFN)={np.corrcoef(ot[m],op[m])[0,1]:.3f}  corr(GBM,TabPFN)={np.corrcoef(og[m],op[m])[0,1]:.3f}", flush=True)
    print(f"  0.5GBM+0.5TabICL      {sc(0.5*og+0.5*ot):.4f}", flush=True)
    print(f"  0.5GBM+0.25TabICL+0.25TabPFN {sc(0.5*og+0.25*ot+0.25*op):.4f}", flush=True)
    print(f"  0.4GBM+0.3TabICL+0.3TabPFN   {sc(0.4*og+0.3*ot+0.3*op):.4f}", flush=True)
