import subprocess, sys, glob, shutil, os, numpy as np
os.makedirs("/kaggle/working/wh", exist_ok=True); os.makedirs("/kaggle/working/wt", exist_ok=True)
# 1) download wheels (tabicl + einops), NOT torch (Kaggle already has it)
subprocess.run([sys.executable,"-m","pip","download","--no-deps","tabicl","einops","-d","/kaggle/working/wh"], check=False)
# 2) install to trigger weight download
subprocess.run([sys.executable,"-m","pip","install","-q","--no-deps","tabicl"], check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","einops","huggingface_hub"], check=False)
from tabicl import TabICLRegressor
reg=TabICLRegressor(device="cpu")
try:
    reg.fit(np.random.rand(60,5).astype("float32"), np.random.rand(60).astype("float32"))
    reg.predict(np.random.rand(6,5).astype("float32"))
    print("fit/predict OK -> weights cached")
except Exception as e:
    print("fit err:", repr(e)[:200])
# 3) locate + copy the cached checkpoint(s)
found=[]
for pat in ["/root/.cache/**/*.ckpt", os.path.expanduser("~/.cache/**/*.ckpt"), "/root/.cache/**/*.safetensors", "/tmp/**/*.ckpt"]:
    found += glob.glob(pat, recursive=True)
for f in sorted(set(found)):
    print("ckpt:", f, round(os.path.getsize(f)/1e6,1),"MB"); shutil.copy(f, "/kaggle/working/wt/")
print("WHEELS:", os.listdir("/kaggle/working/wh"))
print("WEIGHTS:", os.listdir("/kaggle/working/wt"))
import tabicl; print("tabicl ver:", getattr(tabicl,"__version__","?"))
