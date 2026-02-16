"""
Verify Additivity of Integrated Gradients.
IG guarantees: sum(attr) == f(x) - f(ref)  (up to numerical precision)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'MalConv2-main'))

import torch
import numpy as np
import importlib
import src.compute_DeepShap as cds
importlib.reload(cds)
from src.compute_DeepShap import MalConvGCTDeepShap
from MalConvGCT_nocat import MalConvGCT

# ─── 1. Load real model ───
device = torch.device('cpu')
ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'MalConv2-main', 'malconvGCT_nocat.checkpoint')

model = MalConvGCTDeepShap(channels=256, window_size=256, stride=64, embd_size=8, out_size=2)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.to(device).eval()
print("[OK] Model loaded")

# ─── 2. Load real sample ───
sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Locky')
with open(sample_path, 'rb') as f:
    raw = f.read()

MAX_LEN = 100000
raw_bytes = np.frombuffer(raw[:MAX_LEN], dtype=np.uint8)
input_tensor = torch.tensor(raw_bytes.astype(np.int32) + 1, dtype=torch.long).unsqueeze(0).to(device)
print(f"[OK] Input shape: {input_tensor.shape}")

# ─── 3. Normal forward (integer) for model output ───
model.saved_indices = None
model.context_net.saved_indices = None

with torch.no_grad():
    logits_normal = MalConvGCT.forward(model, input_tensor)
    if isinstance(logits_normal, tuple):
        logits_normal = logits_normal[0]
model_out = logits_normal[0, 1].item()
print(f"Model output (class 1): {model_out:.6f}")

# ─── 4. Baseline output (zero embeddings) ───
embd_size = model.embd_size
ref_combined = torch.zeros((1, input_tensor.shape[1], 2*embd_size), device=device)

model._is_explaining = True
model.context_net._is_explaining = True
with torch.no_grad():
    ref_logits = model(ref_combined)
base_val = ref_logits[0, 1].item()
model._is_explaining = False
model.context_net._is_explaining = False
print(f"Baseline output (class 1): {base_val:.6f}")

# ─── 5. Reset saved_indices & recompute ───
model.saved_indices = None
model.context_net.saved_indices = None
with torch.no_grad():
    _ = MalConvGCT.forward(model, input_tensor)

# ─── 6. Explain with IG (n_steps=50) ───
import time
print("\n--- Running Integrated Gradients (n_steps=50) ---")
t0 = time.time()
shap_ctx, shap_feat = model.explainer.explain(input_tensor, target_class=1)
elapsed = time.time() - t0
print(f"Time: {elapsed:.1f}s")

shap_sum = shap_ctx.sum() + shap_feat.sum()
expected = model_out - base_val
diff = abs(shap_sum - expected)

print(f"\n=== ADDITIVITY CHECK ===")
print(f"SHAP Sum (ctx+feat): {shap_sum:.6f}")
print(f"f(x) - f(ref):       {expected:.6f}")
print(f"Diff:                 {diff:.6f}")
print(f"Context |max|:        {np.abs(shap_ctx).max():.6f}")
print(f"Feature |max|:        {np.abs(shap_feat).max():.6f}")
print(f"Context sum:          {shap_ctx.sum():.6f}")
print(f"Feature sum:          {shap_feat.sum():.6f}")

if diff < 0.5:
    print("\n✅ ADDITIVITY PASSED!")
elif diff < 2.0:
    print(f"\n⚠️  ADDITIVITY CLOSE (Diff={diff:.4f})")
else:
    print(f"\n❌ ADDITIVITY FAILED (Diff={diff:.2f})")
