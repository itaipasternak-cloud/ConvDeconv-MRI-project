"""CPU end-to-end test (see README.md). Execute the notebook the way a SLURM job does (tagged cells stripped, CD_* env overrides),
on CPU with a shrunken network. Usage: run_nb.py <repo> <ckpt_root> <data_dir> KEY=VAL ..."""
import json, os, sys, time
repo, ckpt, data = sys.argv[1:4]
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1); os.environ[k] = v
os.environ.update(CONVDECODER_CKPT_ROOT=ckpt, CONVDECODER_DATA_DIR=data, MPLBACKEND="Agg", TQDM_DISABLE="1")
import torch
torch.cuda.FloatTensor = torch.FloatTensor          # CPU stand-in for the GPU tensor type
from IPython.core.inputtransformer2 import TransformerManager
tf = TransformerManager().transform_cell
nb = json.load(open(f"{repo}/MRI_ConvDeconv_variance_earlystop.ipynb"))
ns = {"__name__": "__main__", "REPO_DIR": repo, "IN_COLAB": False, "display": print}
sys.path.insert(0, repo); os.chdir(repo)
t0 = time.time()
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    if {"skip-slurm", "no-unattended"} & set(c.get("metadata", {}).get("tags", [])): continue
    s = "".join(c["source"])
    if "REPO_SLUG" in s: continue                                    # git pull / pip / ssh check
    s = s.replace("num_channels = 256", "num_channels = 32")           # CPU-sized network
    try:
        exec(compile(tf(s), f"<cell {i}>", "exec"), ns)
    except Exception:
        print(f"\n!!! FAILED in cell {i}"); raise
print(f"\nE2E OK in {time.time() - t0:.0f}s")
