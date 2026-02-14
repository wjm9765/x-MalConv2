# MalConvGCTDeepShap Adversarial Attack & Comparison Test
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib
import shap
from matplotlib.lines import Line2D

# 1. Path Setup (MUST be done before imports)
current_dir = os.getcwd()

# (1) Ensure src is in path
if os.path.exists('src'):
    sys.path.append(os.path.abspath('src'))
    sys.path.append(os.getcwd())
elif os.path.exists('../src'):
    sys.path.append(os.path.abspath('../src'))
    sys.path.append(os.path.abspath('..'))
# Ensure we can import from parent directory if running from test/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# (2) MalConv2-main Path Setup (Critical for MalConvGCT_nocat import)
possible_paths = [
    os.path.join(current_dir, '../models/MalConv2-main'),
    os.path.join(current_dir, 'models/MalConv2-main'),
    '/Users/wjm/Desktop/2026 프로젝트/Binary-Hunter/models/MalConv2-main',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/MalConv2-main'))
]

malconv_path = None
for p in possible_paths:
    if os.path.exists(p) and os.path.isdir(p):
        malconv_path = os.path.abspath(p)
        break

if malconv_path:
    if malconv_path not in sys.path: 
        sys.path.append(malconv_path)
        print(f"MalConv Path Added: {malconv_path}")
else:
    print("Warning: MalConv2-main path not found. Imports may fail.")


# 2. Imports (Now that paths are set)
try:
    from src.compute_DeepShap import MalConvGCTDeepShap
    from src.adversarial_malware import generate_adversarial_example
    from src import preprocess_pe_file
    from src.utils import load_config
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    raise e

print("\n--- Testing Adversarial Attack & DeepSHAP Comparison ---")

# Load Config to check settings
config = load_config()
nsamples = config.get('shap_nsamples', 10)
fast_approx = config.get('shap_fast_approximation', False)
print(f"SHAP Configuration: FastApprox={fast_approx}, nsamples={nsamples}")
if fast_approx:
    print(">> Using Fast Gradient Approximation (Input * Gradient). Should be extremely fast.")

def create_new_model():
    """Factory for fresh model instances"""
    channels = 256
    window_size = 256
    stride = 64
    embd_size = 8
    
    model = MalConvGCTDeepShap(out_size=2, channels=channels, window_size=window_size, stride=stride, embd_size=embd_size)
    
    if malconv_path:
        checkpoint_path = os.path.join(malconv_path, 'malconvGCT_nocat.checkpoint')
        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
    model.eval()
    return model

# Setup Data
data_dir_candidates = [
    os.path.join(current_dir, '../data'),
    os.path.join(current_dir, 'data'),
    '/Users/wjm/Desktop/2026 프로젝트/Binary-Hunter/data'
]
data_dir = None
for d in data_dir_candidates:
    if os.path.exists(d) and os.path.isdir(d):
        data_dir = d
        break

if not data_dir:
    print("Data directory not found. Using dummy data if needed.")
    files_to_process = []
else:
    files_to_process = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    files_to_process = files_to_process[:7]
    print(f"Target Files (Max 7): {files_to_process}")

def plot_discrete_shap(ax, data, title, threshold=1e-5):
    if data.ndim > 1: data = data.flatten()
    active_mask = np.abs(data) > threshold
    indices = np.where(active_mask)[0]
    values = data[indices]
    
    if len(indices) == 0:
        ax.text(0.5, 0.5, "No significant contributions", ha='center', va='center', transform=ax.transAxes)
        # ax.set_title(title)
        return

    colors = ['red' if v > 0 else 'blue' for v in values]
    ax.vlines(indices, 0, values, colors=colors, linewidth=1.0, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    # ax.set_title(title)
    ax.set_ylabel('Impact')
    ax.grid(True, alpha=0.2, axis='y')

def verify_shap_additivity(model, shap_vals_sum, input_len, logic_logit, description):
    device = next(model.parameters()).device
    embd_size = model.embd_size
    baseline_combined = torch.zeros((1, input_len, 2*embd_size)).to(device)
    
    model._is_explaining = True
    if hasattr(model, "context_net"): model.context_net._is_explaining = True
        
    try:
        with torch.no_grad():
            base_logits = model(baseline_combined)
            base_value = base_logits[0, 1].item()
    finally:
        model._is_explaining = False
        if hasattr(model, "context_net"): model.context_net._is_explaining = False
            
    reconstructed = shap_vals_sum + base_value
    diff = abs(reconstructed - logic_logit)
    
    print(f"  [{description}] Verification:")
    print(f"    SHAP Sum : {shap_vals_sum:.4f}")
    print(f"    Base Val : {base_value:.4f}")
    print(f"    Sum+Base : {reconstructed:.4f}")
    print(f"    Model Out: {logic_logit:.4f}")
    print(f"    Diff     : {diff:.6f}")
    
    if diff < 1.0:
        print("    >> PASS: Additivity Verified.")
    else:
        print("    >> FAIL: Additivity Mismatch.")

# Limit processing to first file for quick test
if len(files_to_process) > 1:
    print("Optimization: Processing ONLY the first file to test speed.")
    files_to_process = files_to_process[:1]

for idx, filename in enumerate(files_to_process):
    file_path = os.path.join(data_dir, filename)
    print(f"\n[{idx+1}/{len(files_to_process)}] Processing: {filename}")
    
    try:
        with open(file_path, 'rb') as f:
            raw_bytes = f.read(4000000) 
        orig_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
        
        # Original
        model_orig = create_new_model()
        orig_tensor = torch.tensor(orig_bytes.astype(np.int32) + 1, dtype=torch.long).unsqueeze(0)
        
        if hasattr(model_orig, "saved_indices"): model_orig.saved_indices = None
        if hasattr(model_orig.context_net, "saved_indices"): model_orig.context_net.saved_indices = None
        
        print("  Running Original Inference & SHAP...")
        out_orig = model_orig(orig_tensor)
        logits_orig = out_orig[0]
        prob_orig = torch.nn.functional.softmax(logits_orig, dim=1)[0, 1].item()
        logit_orig_val = logits_orig[0, 1].item()
        shap_ctx_orig, shap_feat_orig = out_orig[3], out_orig[4]
        
        verify_shap_additivity(model_orig, np.sum(shap_ctx_orig) + np.sum(shap_feat_orig), len(orig_bytes), logit_orig_val, "Original")
        
        # Adversarial
        print("  Generating Adversarial Example...")
        adv_bytes = generate_adversarial_example(model_orig, orig_bytes, target_class=0)
        del model_orig
        
        # Adversarial Analysis
        model_adv = create_new_model()
        adv_tensor = torch.tensor(adv_bytes.astype(np.int32) + 1, dtype=torch.long).unsqueeze(0)
        
        if hasattr(model_adv, "saved_indices"): model_adv.saved_indices = None
        if hasattr(model_adv.context_net, "saved_indices"): model_adv.context_net.saved_indices = None

        print("  Running Adversarial Inference & SHAP...")
        out_adv = model_adv(adv_tensor)
        logits_adv = out_adv[0]
        prob_adv = torch.nn.functional.softmax(logits_adv, dim=1)[0, 1].item()
        logit_adv_val = logits_adv[0, 1].item()
        shap_ctx_adv, shap_feat_adv = out_adv[3], out_adv[4]
        
        print(f"  Result: Original Prob={prob_orig*100:.2f}% -> Adversarial Prob={prob_adv*100:.2f}%")
        verify_shap_additivity(model_adv, np.sum(shap_ctx_adv) + np.sum(shap_feat_adv), len(adv_bytes), logit_adv_val, "Adversarial")

        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        vis_threshold = 1e-5
        
        # Titles
        axes[0].set_title(f'Original - Context (Prob: {prob_orig:.4f})')
        axes[1].set_title('Original - Feature')
        axes[2].set_title(f'Adversarial - Context (Prob: {prob_adv:.4f})')
        axes[3].set_title('Adversarial - Feature')

        plot_discrete_shap(axes[0], shap_ctx_orig, '', vis_threshold)
        plot_discrete_shap(axes[1], shap_feat_orig, '', vis_threshold)
        plot_discrete_shap(axes[2], shap_ctx_adv, '', vis_threshold)
        plot_discrete_shap(axes[3], shap_feat_adv, '', vis_threshold)
        
        total_len = len(adv_bytes)
        axes[3].set_xlim(0, total_len)
        custom_lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='blue', lw=2)]
        fig.legend(custom_lines, ['Increases Malware Score', 'Decreases Malware Score'], loc='upper right')
        axes[3].set_xlabel('Byte Index from Start of File')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        output_plot_path = f"shap_result_{filename}.png"
        plt.savefig(output_plot_path)
        print(f"Plot saved to {output_plot_path}")
        plt.close()
        
        del model_adv

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()

print("\nComparison Run Complete.")
