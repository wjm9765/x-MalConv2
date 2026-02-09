import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils import load_config
from contextlib import contextmanager
import sys

# Try to import MalConvGCT. If strictly following "no modification to models", we assume it's importable.
# If PYTHONPATH is not set, we add it.
try:
    from MalConvGCT_nocat import MalConvGCT
except ImportError:
    sys.path.append('models/MalConv2-main')
    from MalConvGCT_nocat import MalConvGCT

class DeepShapExplainer:
    def __init__(self, model):
        self.model = model
        self.config = load_config()
        # Default to 'zero' if config missing
        self.baseline_type = self.config.get('explainability', {}).get('deep_shap', {}).get('baseline', 'zero')
        
        self.captured_embeddings = []
        
        from LowMemConv import LowMemConvBase
        self.original_seq2fix = LowMemConvBase.seq2fix

    def _get_baseline_input(self, input_tensor):
        if self.baseline_type == 'zero':
            return torch.zeros_like(input_tensor)
        else:
            return torch.zeros_like(input_tensor)

    @staticmethod
    def _custom_seq2fix(self_instance, x, pr_args={}):
        """
        Monkey Patch for LowMemConvBase.seq2fix to capture indices and embeddings (indirectly).
        """
        receptive_window, stride, out_channels = self_instance.determinRF()
        
        if x.shape[1] < receptive_window: 
            x = F.pad(x, (0, receptive_window-x.shape[1]), value=0)
        
        batch_size = x.shape[0]
        length = x.shape[1]
        
        winner_values = np.zeros((batch_size, out_channels))-1.0
        winner_indices = np.zeros((batch_size, out_channels), dtype=np.int64)
            
        if not hasattr(self_instance, "device_ids"):
            cur_device = next(self_instance.embd.parameters()).device
        else:
            cur_device = None

        step = self_instance.chunk_size 
        start = 0
        end = start+step
        
        with torch.no_grad():
            while start < end and (end-start) >= max(self_instance.min_chunk_size, receptive_window):
                x_sub = x[:,start:end]
                if cur_device is not None:
                    x_sub = x_sub.to(cur_device)
                activs = self_instance.processRange(x_sub.long(), **pr_args)
                activ_win, activ_indx = F.max_pool1d(activs, kernel_size=activs.shape[2], return_indices=True)
                
                activ_win = activ_win.cpu().numpy()[:,:,0]
                activ_indx = activ_indx.cpu().numpy()[:,:,0]
                selected = winner_values < activ_win
                winner_indices[selected] = activ_indx[selected]*stride + start 
                winner_values[selected]  = activ_win[selected]
                start = end
                end = min(start+step, length)

        final_indices = [np.unique(winner_indices[b,:]) for b in range(batch_size)]
        
        chunk_list = [[x[b:b+1,max(i-receptive_window,0):min(i+receptive_window,length)] for i in final_indices[b]] for b in range(batch_size)]
        chunk_list = [torch.cat(c, dim=1)[0,:] for c in chunk_list]
        
        x_selected = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True)
        
        # Capture indices
        if not hasattr(self_instance, '_shap_captured_indices'):
            self_instance._shap_captured_indices = []
        self_instance._shap_captured_indices.append(final_indices)
        
        if cur_device is not None:
            x_selected = x_selected.to(cur_device)
            
        output = self_instance.processRange(x_selected.long(), **pr_args)
        
        output = self_instance.pooling(output)
        output = output.view(output.size(0), -1)
            
        return output

    @contextmanager
    def _patch_model(self):
        from LowMemConv import LowMemConvBase
        original = LowMemConvBase.seq2fix
        LowMemConvBase.seq2fix = self._custom_seq2fix
        try:
            yield
        finally:
            LowMemConvBase.seq2fix = original

    def explain(self, input_tensor, target_class=1):
        """
        Compute Deep SHAP values for Context and Feature separately.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Temporarily disable low_mem to avoid re-computation and duplicate index capturing
        # Check if attribute exists (MalConvGCT has it, MalConvML inherits but effectively LowMemConvBase doesn't have it, MalConvGCT adds it)
        original_low_mem = getattr(self.model, 'low_mem', False)
        if hasattr(self.model, 'low_mem'):
            self.model.low_mem = False

        ref_input = self._get_baseline_input(input_tensor).to(input_tensor.device)
        input_tensor = input_tensor.to(input_tensor.device)

        self.model._shap_captured_indices = []
        self.captured_embeddings = []

        def embedding_forward_hook(module, input, output):
            # Only capture if gradient calculation is enabled and required
            # This filters out the forward passes during the 'max-pooling search' phase (inside torch.no_grad())
            if torch.is_grad_enabled() and output.requires_grad:
                output.retain_grad()
                self.captured_embeddings.append(output)

        handle = self.model.embd.register_forward_hook(embedding_forward_hook)
        
        handles = [handle]
        # Register hook for context_net.embd if distinct
        if hasattr(self.model, 'context_net') and hasattr(self.model.context_net, 'embd'):
            if self.model.context_net.embd is not self.model.embd:
                h2 = self.model.context_net.embd.register_forward_hook(embedding_forward_hook)
                handles.append(h2)

        shap_maps = []
        try:
            with self._patch_model():
                outputs = self.model(input_tensor)
                logits = outputs[0]
                
                # Default to target class 1 or 0 if unary
                if logits.shape[1] > 1:
                     target_score = logits[0, target_class]
                else:
                     target_score = logits[0, 0]
                
                self.model.zero_grad()
                target_score.backward()
                
                total_len = input_tensor.shape[1]
                
                embeddings_list = self.captured_embeddings
                indices_list = self.model._shap_captured_indices
                
                with torch.no_grad():
                    ref_emb_vec = self.model.embd(torch.tensor([0]).to(input_tensor.device))
                
                # Expecting 2 passes: context and main
                for i in range(len(embeddings_list)):
                    if i >= len(indices_list): break
                    
                    emb_out = embeddings_list[i]
                    indices_batch = indices_list[i]
                    
                    if emb_out.grad is None:
                        # Append zero map if no grad
                        shap_maps.append(np.zeros(total_len))
                        continue
                        
                    grad = emb_out.grad
                    diff = emb_out - ref_emb_vec
                    shap_per_token = (grad * diff).sum(dim=-1) # (B, L_chunk)
                    
                    shap_values_np = shap_per_token.detach().cpu().numpy()[0]
                    
                    final_shap_map = np.zeros(total_len)
                    real_indices = indices_batch[0]
                    
                    current_ptr = 0
                    
                    # Determine RF for reconstruction
                    if i == 0 and hasattr(self.model, 'context_net'):
                        rf, _, _ = self.model.context_net.determinRF()
                    else:
                        rf, _, _ = self.model.determinRF()
                        
                    for start_idx in real_indices:
                        s = max(start_idx - rf, 0)
                        e = min(start_idx + rf, total_len)
                        if e > total_len: e = total_len
                        chunk_len = e - s
                        
                        if current_ptr + chunk_len <= len(shap_values_np):
                            shap_chunk = shap_values_np[current_ptr : current_ptr + chunk_len]
                            final_shap_map[s:e] += shap_chunk
                            current_ptr += chunk_len
                        else:
                            # If size mismatch (e.g. padding/chunks), break or fill
                            # Usually exact match if logic is correct
                            pass
                    
                    shap_maps.append(final_shap_map)

        finally:
            for h in handles:
                h.remove()
            if hasattr(self.model, 'low_mem'):
                self.model.low_mem = original_low_mem
            if hasattr(self.model, '_shap_captured_indices'):
                del self.model._shap_captured_indices
        
        return shap_maps

class MalConvGCTDeepShap(MalConvGCT):
    def __init__(self, out_size=2, channels=128, window_size=512, stride=512, layers=1, embd_size=8, log_stride=None, low_mem=True):
        # Match MalConvGCT constructor to ensure compatibility
        super(MalConvGCTDeepShap, self).__init__(out_size=out_size, channels=channels, window_size=window_size, stride=stride, layers=layers, embd_size=embd_size, log_stride=log_stride, low_mem=low_mem)
        self.explainer = DeepShapExplainer(self)
        self._is_explaining = False

    def forward(self, x):
        # If explaining, run standard forward (explainer will handle hooks)
        if self._is_explaining:
            return super(MalConvGCTDeepShap, self).forward(x)
        
        # Normal execution
        # Get standard outputs
        outputs = super(MalConvGCTDeepShap, self).forward(x)
        
        # Compute SHAP
        self._is_explaining = True
        try:
            # We assume we want to explain the prediction of the current forward pass
            # Explain Class 1 (Malware) by default or highest prob? 
            # Usually for malware detection, we explain "Why is it malware?" (Class 1)
            target = 1 
            
            shap_maps = self.explainer.explain(x, target_class=target)
            
            if len(shap_maps) >= 2:
                # Assuming order: Context, Feature (from sequential execution)
                shap_context = shap_maps[0]
                shap_feature = shap_maps[1]
            elif len(shap_maps) == 1:
                shap_context = shap_maps[0]
                shap_feature = np.zeros_like(shap_maps[0])
            else:
                shap_context = np.zeros(x.shape[1])
                shap_feature = np.zeros(x.shape[1])
                
        finally:
            self._is_explaining = False
            
        # Return extended output
        return outputs + (shap_context, shap_feature)

def compute_deep_shap(model, input_tensor, target_class=1):
    explainer = DeepShapExplainer(model)
    return explainer.explain(input_tensor, target_class)
