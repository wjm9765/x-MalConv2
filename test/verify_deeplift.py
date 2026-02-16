"""Minimal test: verify Context attribution is non-zero with DeepLIFT pooling fix."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/MalConv2-main')))

import torch, numpy as np, importlib
import src.compute_DeepShap
importlib.reload(src.compute_DeepShap)
from src.compute_DeepShap import MalConvGCTDeepShap

# Small model for fast test
model = MalConvGCTDeepShap(out_size=2, channels=16, window_size=32, stride=8, embd_size=8)
model.eval()

dummy = torch.randint(1, 256, (1, 2000)).long()

print("1) Inference ...")
with torch.no_grad():
    out = model(dummy)
print(f"   logits = {out[0]}")

print("2) SHAP explain ...")
result = model.explainer.explain(dummy, target_class=1)
ctx, feat = result

print(f"   Context shape={ctx.shape}  sum={np.sum(ctx):.6f}  absmax={np.max(np.abs(ctx)):.6f}")
print(f"   Feature shape={feat.shape} sum={np.sum(feat):.6f}  absmax={np.max(np.abs(feat)):.6f}")

if np.max(np.abs(ctx)) > 1e-8:
    print("\n   ✅ SUCCESS: Context attribution is NON-ZERO")
else:
    print("\n   ❌ FAIL: Context attribution is still zero")

if np.max(np.abs(feat)) > 1e-8:
    print("   ✅ SUCCESS: Feature attribution is NON-ZERO")
else:
    print("   ❌ FAIL: Feature attribution is zero")
