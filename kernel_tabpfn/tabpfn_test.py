# TabPFN vs LGB+Cat on OUR 37 features (subsampled), lockbox RMSE. Experiment (GPU+internet).
import subprocess, sys, numpy as np
subprocess.run([sys.executable,"-m","pip","install","-q","tabpfn"], check=False)
import glob
_f=glob.glob("/kaggle/input/**/tabpfn_data.npz",recursive=True)
print("found:",_f)
d = np.load(_f[0])
Xc,yc,Xl = d["X_ctx"], d["y_ctx"], d["X_lock"]
truth,anchor,fusion = d["truth_lock"], d["anchor_lock"], d["fusion_lock"]
def rmse(p): return float(np.sqrt(np.mean((truth - (0.75*(anchor+p)+0.25*fusion))**2)))
print(f"context {Xc.shape}, lock {Xl.shape}", flush=True)

# baseline: LGB+Cat on the SAME subsample
import lightgbm as lgb
from catboost import CatBoostRegressor
lg = lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=63,min_child_samples=200,
        reg_lambda=1.0,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,random_state=0,n_jobs=-1,verbose=-1).fit(Xc,yc)
cb = CatBoostRegressor(iterations=600,learning_rate=0.03,depth=8,l2_leaf_reg=3.0,rsm=0.8,random_seed=0,thread_count=-1,verbose=0).fit(Xc,yc)
gbm = 0.5*(lg.predict(Xl)+cb.predict(Xl))
print(f"LGB+Cat (same subsample):  lockbox {rmse(gbm):.4f}", flush=True)

# TabPFN
try:
    from tabpfn import TabPFNRegressor
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = min(len(Xc), 20000)  # TabPFN context cap
    idx = np.random.default_rng(0).choice(len(Xc), n, replace=False)
    reg = TabPFNRegressor(device=dev)
    reg.fit(Xc[idx], yc[idx])
    tp = reg.predict(Xl)
    print(f"TabPFN (ctx {n}, {dev}):      lockbox {rmse(tp):.4f}", flush=True)
    print(f"mean(TabPFN,LGB+Cat):      lockbox {rmse(0.5*(tp+gbm)):.4f}", flush=True)
except Exception as e:
    print("TabPFN failed:", repr(e), flush=True)
