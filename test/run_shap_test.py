# MalConvGCTDeepShap Adversarial Attack & Comparison Test
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib
import shap
from matplotlib.lines import Line2D

print("\n--- Testing Adversarial Attack & DeepSHAP Comparison ---")

# 1. Path Setup
current_dir = os.getcwd()
src_path = os.path.abspath(os.path.join(current_dir, '../src')) # Changed to point to src folder directly if needed, or adjust relative path
# Assuming the script is run from 'test/' directory or root. Let's make it robust.

# If running from root: current_dir is root. src is in 'src'.
# If running from test: current_dir is test. src is in '../src'.
if os.path.exists('src'):
    sys.path.append(os.path.abspath('src'))
    sys.path.append(os.getcwd()) # Add root to path
elif os.path.exists('../src'):
    sys.path.append(os.path.abspath('../src'))
    sys.path.append(os.path.abspath('..')) # Add root to path

# MalConv2-main Path
possible_paths = [
    os.path.join(current_dir, '../models/MalConv2-main'),
    os.path.join(current_dir, 'models/MalConv2-main'),
    '/Users/wjm/Desktop/2026 프로젝트/Binary-Hunter/models/MalConv2-main'
]

malconv_path = None
for p in possible_paths:
    if os.path.exists(p) and os.path.isdir(p):
        malconv_path = os.path.abspath(p)
        break

if malconv_path:
    if malconv_path not in sys.path: sys.path.append(malconv_path)
    print(f"MalConv2-main path added: {malconv_path}")
else:
    print("Warning: MalConv2-main path not found.")

# Import Modules
try:
    from src.compute_DeepShap import MalConvGCTDeepShap
    from src.adversarial_malware import generate_adversarial_example
    from src import preprocess_pe_file
except ImportError as e:
    # Fallback import if running from inside test folder and sys.path setup is tricky
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.compute_DeepShap import MalConvGCTDeepShap
        from src.adversarial_malware import generate_adversarial_example
        from src import preprocess_pe_file
    except Exception as e2:
        print(f"Import failed: {e}")
        print(f"Fallback failed: {e2}")
        sys.exit(1)

# Helper: Initialize New Model (Factory Function)
def create_new_model():
    """
    Creates a fresh instance of MalConvGCTDeepShap to avoid any state leakage.
    """
    channels = 256
    window_size = 256
    stride = 64
    embd_size = 8
    
    model = MalConvGCTDeepShap(out_size=2, channels=channels, window_size=window_size, stride=stride, embd_size=embd_size)
    
    if malconv_path:
        checkpoint_path = os.path.join(malconv_path, 'malconvGCT_nocat.checkpoint')
        if os.path.exists(checkpoint_path):
            try:
                # Map location to CPU for compatibility
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
                print("Checkpoint loaded.")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                
    model.eval()
    return model

# 4. Data Setup
data_dir_candidates = [
    os.path.join(current_dir, '../data'),
    os.path.join(current_dir, 'data'),
    '/Users/wjm/Desktop/2026 프로젝트/Binary-Hunter/data'
]
data_dir = None
for d in data_dir_candidates:
    if os.path.exists(d) and os.path.isdir(d):
        data_dir = d
        print(f"Data directory found: {data_dir}")
        break

if not data_dir:
    print("Data directory not found!")
    files_to_process = []
else:
    files_to_process = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    files_to_process = files_to_process[:7] # Limit to 7 files
    print(f"Target Files (Max 7): {files_to_process}")

# Helper: Discrete Plot
def plot_discrete_shap(ax, data, title, threshold=1e-5):
    if data.ndim > 1: data = data.flatten()
    
    # Filter by threshold
    active_mask = np.abs(data) > threshold
    indices = np.where(active_mask)[0]
    values = data[indices]
    
    if len(indices) == 0:
        ax.text(0.5, 0.5, "No significant contributions", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    colors = ['red' if v > 0 else 'blue' for v in values]
    
    ax.vlines(indices, 0, values, colors=colors, linewidth=1.0, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    
    ax.set_title(title)
    ax.set_ylabel('Impact')
    ax.grid(True, alpha=0.2, axis='y')

# Helper: Verification
def verify_shap_additivity(model, shap_vals_sum, input_len, logic_logit, description):
    """
    Verifies that Sum(SHAP) + BaseValue == ModelOutput.
    BaseValue must be calculated using ZeroEmbeddings passed through the FIXED graph (saved indices).
    """
    device = next(model.parameters()).device
    
    # 1. Prepare Baseline (Zero Embeddings)
    # DeepSHAP uses Zero Embeddings as baseline
    embd_size = model.embd_size
    baseline_combined = torch.zeros((1, input_len, 2*embd_size)).to(device)
    
    # 2. Compute Base Value with FORCED Indices
    # We must enable _is_explaining on both main and context models to use saved_indices
    model._is_explaining = True
    if hasattr(model, "context_net"):
        model.context_net._is_explaining = True
        
    try:
        with torch.no_grad():
            base_logits = model(baseline_combined)
            # The model output is raw Logits.
            # We are verifying Class 1 (Malware) Logit.
            base_value = base_logits[0, 1].item()
    finally:
        model._is_explaining = False
        if hasattr(model, "context_net"):
            model.context_net._is_explaining = False
            
    # 3. Compare
    # SHAP Sum represents the change from Expected Value (Base Value) to Output
    # So: Output ≈ BaseValue + Sum(SHAP)
    reconstructed = shap_vals_sum + base_value
    diff = abs(reconstructed - logic_logit)
    
    print(f"  [{description}] Verifiction:")
    print(f"    SHAP Sum : {shap_vals_sum:.4f}")
    print(f"    Base Val : {base_value:.4f} (Model output on Zero Embeddings with Fixed Indices)")
    print(f"    Sum+Base : {reconstructed:.4f}")
    print(f"    Model Out: {logic_logit:.4f}")
    print(f"    Diff     : {diff:.6f}")
    
    if diff < 1.0:
        print("    >> PASS: Additivity Verified.")
    else:
        print("    >> FAIL: Additivity Mismatch.")
        

# 5. Process Loop
for idx, filename in enumerate(files_to_process):
    file_path = os.path.join(data_dir, filename)
    print(f"\n[{idx+1}/{len(files_to_process)}] Processing: {filename}")
    
    try:
        # Read Original Bytes
        with open(file_path, 'rb') as f:
            raw_bytes = f.read(4000000) # Max 4MB 
        orig_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
        
        # --- PHASE 1: Original Sample Analysis ---
        # Create FRESH model instance for Original
        model_orig = create_new_model()
        
        orig_tensor = torch.tensor(orig_bytes.astype(np.int32) + 1, dtype=torch.long).unsqueeze(0)
        
        # Inference & SHAP (Original)
        # 1. Clear saved indices to force fresh computation
        if hasattr(model_orig, "saved_indices"): model_orig.saved_indices = None
        if hasattr(model_orig.context_net, "saved_indices"): model_orig.context_net.saved_indices = None
        
        out_orig = model_orig(orig_tensor)
        logits_orig = out_orig[0]
        prob_orig = torch.nn.functional.softmax(logits_orig, dim=1)[0, 1].item()
        logit_orig_val = logits_orig[0, 1].item()
        shap_ctx_orig, shap_feat_orig = out_orig[3], out_orig[4]
        
        # Verify Original
        shap_sum_orig = np.sum(shap_ctx_orig) + np.sum(shap_feat_orig)
        verify_shap_additivity(model_orig, shap_sum_orig, len(orig_bytes), logit_orig_val, "Original")
        
        # --- PHASE 2: Attack Generation ---
        # Generate Adversarial Example using the Original Model
        # print("  Generating Adversarial Example...")
        adv_bytes = generate_adversarial_example(model_orig, orig_bytes, target_class=0)
        
        # Clean up model_orig to free memory/state
        del model_orig
        
        # --- PHASE 3: Adversarial Sample Analysis ---
        # Create FRESH model instance for Adversarial
        model_adv = create_new_model()
        
        adv_tensor = torch.tensor(adv_bytes.astype(np.int32) + 1, dtype=torch.long).unsqueeze(0)
        
        # Inference & SHAP (Adversarial)
        # 1. Clear saved indices to force fresh computation
        if hasattr(model_adv, "saved_indices"): model_adv.saved_indices = None
        if hasattr(model_adv.context_net, "saved_indices"): model_adv.context_net.saved_indices = None

        out_adv = model_adv(adv_tensor)
        logits_adv = out_adv[0]
        prob_adv = torch.nn.functional.softmax(logits_adv, dim=1)[0, 1].item()
        logit_adv_val = logits_adv[0, 1].item()
        shap_ctx_adv, shap_feat_adv = out_adv[3], out_adv[4]
        
        print(f"  Result: Original Prob={prob_orig*100:.2f}% -> Adversarial Prob={prob_adv*100:.2f}%")
        
        # Verify Adversarial
        shap_sum_adv = np.sum(shap_ctx_adv) + np.sum(shap_feat_adv)
        verify_shap_additivity(model_adv, shap_sum_adv, len(adv_bytes), logit_adv_val, "Adversarial")


        # 5. Visualization (Discrete Style)
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        
        # Filter significant values for cleaner plot
        vis_threshold = 1e-5
        
        plot_discrete_shap(axes[0], shap_ctx_orig, f'Original - Context (Prob: {prob_orig:.4f})', vis_threshold)
        plot_discrete_shap(axes[1], shap_feat_orig, 'Original - Feature', vis_threshold)
        plot_discrete_shap(axes[2], shap_ctx_adv, f'Adversarial - Context (Prob: {prob_adv:.4f})', vis_threshold)
        plot_discrete_shap(axes[3], shap_feat_adv, 'Adversarial - Feature', vis_threshold)
        
        # Force X-Axis to show full file (Adversarial Length)
        total_len = len(adv_bytes)
        axes[3].set_xlim(0, total_len)
        
        custom_lines = [Line2D([0], [0], color='red', lw=2),
                        Line2D([0], [0], color='blue', lw=2)]
        fig.legend(custom_lines, ['Increases Malware Score', 'Decreases Malware Score'], loc='upper right')
        
        axes[3].set_xlabel('Byte Index from Start of File')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save plot to file instead of plt.show()
        output_plot_path = f"shap_result_{filename}.png"
        plt.savefig(output_plot_path)
        print(f"Plot saved to {output_plot_path}")
        plt.close() # Close figure to free memory
        
        # Cleanup
        del model_adv

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()

print("\nComparison Complete.")
