"""Design-level deployment numbers for the trained actor: parameter count, MACs, serialized sizes,
NumPy inference latency on the study CPU, and the action error of int8 post-training quantization."""
import sys, os, json, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from evaluate import LoadedActor

out = {}
f = sorted(glob.glob('results/models/pacac_s*_actor.npz'))[0]
A = LoadedActor(f)
n_par = sum(W.size + b.size for W, b in A.W)
macs = sum(W.size for W, b in A.W)
out.update(params=int(n_par), macs=int(macs), fp32_bytes=int(n_par * 4),
           int8_bytes=int(n_par + 4 * len(A.W) * 2), layers=[list(W.shape) for W, b in A.W])
od = A.W[0][0].shape[1]
X = np.random.default_rng(0).normal(0, 1, (20000, od)).astype(np.float32)
t0 = time.perf_counter()
for x in X[:10000]:
    A(x)
out['numpy_latency_us'] = (time.perf_counter() - t0) / 10000 * 1e6


def quant(W):
    s = np.abs(W).max() / 127.0
    return np.round(W / s).astype(np.int8), s


Aq = LoadedActor(f)
Aq.W = [(quant(W)[0].astype(np.float32) * quant(W)[1], b) for W, b in A.W]
d = np.array([abs(A(x) - Aq(x)) for x in X[10000:]])
out['int8_max_abs_err_mm'] = float(d.max()); out['int8_mean_abs_err_mm'] = float(d.mean())
out['transition_record_bytes'] = 32; out['upload_B_per_h'] = 64
out['update_B_per_week'] = out['int8_bytes'] + 64; out['update_B_per_h'] = out['update_B_per_week'] / 168
json.dump(out, open('results/deploy.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
