import torch
import torch.nn.functional as F
import numpy as np
import shap
from .utils import load_config, log
from MalConvGCT_nocat import MalConvGCT


class DeepShapExplainer:
    """
    Implements DeepSHAP explanation using the official `shap` library.
    
    This wrapper bridges the MalConvGCT model (which takes integer bytes) 
    with shap.GradientExplainer (which is more robust to custom layers like AdaptiveMaxPool1d).
    Replaces DeepExplainer with GradientExplainer to resolve Additivity Mismatch and Layer Recognition issues.
    """

    def __init__(self, model):
        self.model = model
        self.config = load_config()

    def explain(self, input_tensor, target_class=1):
        """
        Compute SHAP values using shap.GradientExplainer.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Disable low_mem to avoid checkpointing issues with SHAP
        original_low_mem = getattr(self.model, "low_mem", False)
        if hasattr(self.model, "low_mem"):
            self.model.low_mem = False

        try:
            input_tensor = input_tensor.to(next(self.model.parameters()).device)
            
            # 1. Prepare Embeddings (Inputs for SHAP)
            # We concatenate ContextNet and FeatureNet embeddings into a single tensor
            # to work around SHAP explainer issues with multiple inputs.
            # This ensures we get a single attribution map (B, L, 2*D) that we can split later.
            with torch.no_grad():
                emb_main = self.model.embd(input_tensor)
                emb_context = self.model.context_net.embd(input_tensor)
                
            # Combine: (B, L, D) + (B, L, D) -> (B, L, 2*D)
            # First half is Context, Second half is Feature
            emb_combined = torch.cat([emb_context, emb_main], dim=-1)

            # 2. Prepare Baseline (Zero Embeddings)
            baseline_combined = torch.zeros_like(emb_combined)
            
            # 3. Initialize SHAP GradientExplainer
            # GradientExplainer is more robust and supports autograd for any layer type
            # It approximates SHAP values by integrating gradients along the path from baseline to input.
            explainer = shap.GradientExplainer(self.model, baseline_combined)
            
            # 4. Compute SHAP Values
            # GradientExplainer returns list/array of attributions
            try:
                # shap_values method for GradientExplainer
                # Reduce nsamples to speed up execution. 
                # Default is 200, which means 200 forward/backward passes per sample.
                # Setting nsamples=50 for faster approximation.
                shap_values = explainer.shap_values(emb_combined, nsamples=50)
            except Exception as e:
                import traceback
                log(f"SHAP explainer.shap_values failed: {e}\n{traceback.format_exc()}", "ERROR")
                raise e
            
            # Debug Log
            val_type = type(shap_values)
            val_len = len(shap_values) if isinstance(shap_values, list) else "N/A"
            # val_shape might not be available on list
            log(f"SHAP Values Info: Type={val_type}, Len={val_len}", "INFO")

            # 5. Extract Target Class Attribution
            target_shap = None
            
            if isinstance(shap_values, list):
                if len(shap_values) <= target_class:
                     if len(shap_values) > 0: target_shap = shap_values[0] # Fallback
                     else: raise IndexError("shap_values is empty list")
                else:
                     target_shap = shap_values[target_class]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 4 and shap_values.shape[-1] > target_class:
                     target_shap = shap_values[..., target_class]
                else:
                     target_shap = shap_values

            if target_shap is None:
                 raise ValueError("Failed to extract target_shap")

            # target_shap should now be ndarray of shape (B, L, 2*D)
            # We split it back into Context and Feature attribution
            try:
                emb_dim = emb_context.shape[-1] # D
                # Verify shape matches combined expectation
                if target_shap.shape[-1] == 2 * emb_dim:
                    attr_context = target_shap[..., :emb_dim]
                    attr_feature = target_shap[..., emb_dim:]
                else:
                    log(f"SHAP Warning: target_shap dim {target_shap.shape[-1]} != 2*emb_dim {2*emb_dim}. Using duplicaton fallback.", "WARNING")
                    attr_context = target_shap
                    attr_feature = target_shap
            except Exception as split_err:
                 log(f"SHAP Split Error: {split_err}. Fallback to duplicate.", "ERROR")
                 attr_context = target_shap
                 attr_feature = target_shap

             # Sum over embedding dimension
            shap_context_map = np.sum(attr_context, axis=-1)[0]
            shap_feature_map = np.sum(attr_feature, axis=-1)[0]
            
            return [shap_context_map, shap_feature_map]

        except Exception as e:
            import traceback
            log(f"SHAP Library Error: {str(e)}\n{traceback.format_exc()}", "ERROR")
            # Fallback or return zeros
            size = input_tensor.shape[1]
            return [np.zeros(size), np.zeros(size)]

        finally:
            # Restore state
             if hasattr(self.model, "low_mem"):
                self.model.low_mem = original_low_mem


class MalConvGCTDeepShap(MalConvGCT):
    """
    Wrapper class for MalConvGCT to enable DeepSHAP explanation with shap library.
    """

    def __init__(
        self,
        out_size=2,
        channels=128,
        window_size=512,
        stride=512,
        layers=1,
        embd_size=8,
        log_stride=None,
        low_mem=True,
    ):
        super().__init__(
            out_size=out_size,
            channels=channels,
            window_size=window_size,
            stride=stride,
            layers=layers,
            embd_size=embd_size,
            log_stride=log_stride,
            low_mem=low_mem,
        )
        self.embd_size = embd_size
        self.explainer = DeepShapExplainer(self)
        self._is_explaining = False
        
        import types
        # 1. Patch seq2fix for both main and context net
        self.seq2fix = types.MethodType(self.__class__.seq2fix, self)
        
        # 2. Patch _process_embeddings (Assign correct version for each model type)
        # Main model is MalConvGCT
        self._process_embeddings = types.MethodType(self.__class__._process_embeddings_gct, self)
        
        if hasattr(self, "context_net"):
            self.context_net.seq2fix = types.MethodType(
                self.__class__.seq2fix, self.context_net
            )
            # Context net is MalConvML (Simpler structure)
            self.context_net._process_embeddings = types.MethodType(
                self.__class__._process_embeddings_ml, self.context_net
            )
            # Initialize _is_explaining and saved_indices for context_net
            self.context_net._is_explaining = False
            self.context_net.saved_indices = None

        # 3. Remove conflicting backward hooks from LowMemConvBase
        # SHAP uses standard backward hooks, and the custom drop_zeros_hook
        # in LowMemConvBase (using register_backward_hook) causes a conflict
        # with PyTorch's newer full backward hooks or SHAP's internal hooks.
        # We remove them to ensure compatibility.
        
        # Remove hook from self.cat
        if hasattr(self, "cat") and hasattr(self.cat, "_backward_hooks"):
            self.cat._backward_hooks.clear()
            
        # Remove hook from context_net.cat if it exists
        if hasattr(self, "context_net") and hasattr(self.context_net, "cat") and hasattr(self.context_net.cat, "_backward_hooks"):
            self.context_net.cat._backward_hooks.clear()

    def _process_embeddings_gct(self, x, gct=None):
        """
        MalConvGCT specific embedding processing (with Gating).
        """
        for conv_glu, linear_cntx, conv_share in zip(self.convs, self.linear_atn, self.convs_share):
            x = F.glu(conv_glu(x), dim=1)
            x = F.leaky_relu(conv_share(x))
            
            B = x.shape[0]
            C = x.shape[1]
            
            if gct is not None:
                ctnx = torch.tanh(linear_cntx(gct))
                ctnx = torch.unsqueeze(ctnx, dim=2)
                x_tmp = x.view(1,B*C,-1) 
                x_tmp = F.conv1d(x_tmp, ctnx, groups=B)
                x_gates = x_tmp.view(B, 1, -1)
                gates = torch.sigmoid( x_gates )
                x = x * gates
        return x

    def _process_embeddings_ml(self, x, gct=None):
        """
        MalConvML specific embedding processing (Simple Conv, No Gating).
        Matches MalConvML.processRange logic.
        """
        # MalConvML uses self.convs and self.convs_1
        # No linear_atn, No GCT gating
        
        for conv_glu, conv_share in zip(self.convs, self.convs_1):
             x = F.leaky_relu(conv_share(F.glu(conv_glu(x), dim=1)))
        
        return x

    def forward(self, x, *args):
        """
        Modified forward to handle both:
        1. Normal inference (Integer/Long input)
        2. SHAP explanation (List/Tuple of Embeddings) or Unpacked arguments
        """
        # Handle SHAP unpacking arguments (x=emb_context, args[0]=emb_main)
        if args:
            x = [x, args[0]]

        # Case 3: Combined Embedding Input (Workaround for SHAP single-input issue)
        # DeepShapExplainer passes (B, L, 2*D) combined tensor.
        if isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[-1] == 2 * self.embd_size:
            # SHAP internal execution - no need to recursively explain
            # Split back into Context and Feature
            
            emb_context = x[..., :self.embd_size]
            emb_main = x[..., self.embd_size:]
            
            # Context Net Path
            global_context = self.context_net.seq2fix(emb_context)
            
            # Feature Net Path
            post_conv = self.seq2fix(emb_main, pr_args={'gct': global_context})
            
            # Rest of the network (FC layers)
            penult = F.leaky_relu(self.fc_1(post_conv))
            logits = self.fc_2(penult)
            return logits

        # IMPORTANT: If input x is FloatTensor (Embeddings) but NOT combined (just 1D embeddings),
        # it likely comes from adversarial attack or gradient calculation.
        # In this case, we DO NOT perform explanation.
        if isinstance(x, torch.Tensor) and x.is_floating_point():
             # Standard Forward for embeddings (using patched seq2fix)
             outputs = super().forward(x)
             # Return just logits (or tuple) without SHAP
             return outputs

        # Case 2: SHAP Execution (Input is list [emb_context, emb_main])
        if isinstance(x, (list, tuple)):
            emb_context, emb_main = x
            
            # Context Net Path
            # We assume seq2fix is patched to handle embeddings
            global_context = self.context_net.seq2fix(emb_context)
            
            # Feature Net Path
            # pr_args passes global context
            # We call self.seq2fix on emb_main
            post_conv = self.seq2fix(emb_main, pr_args={'gct': global_context})
            
            # Rest of the network (FC layers)
            penult = F.leaky_relu(self.fc_1(post_conv))
            logits = self.fc_2(penult)
            return logits

        # Case 1: Normal Inference (or Triggering Explanation)
        # ---------------------------------------------------
        # Optimization: If gradients are disabled (inference mode) or already explaining,
        # we skip the SHAP explanation step. SHAP requires grad access.
        if self._is_explaining or not torch.is_grad_enabled():
            return super().forward(x)

        # Standard Forward
        outputs = super().forward(x)

        # Compute SHAP
        self._is_explaining = True
        if hasattr(self, "context_net"):
            self.context_net._is_explaining = True
            
        try:
            target = 1
            shap_result = self.explainer.explain(x, target_class=target)
            shap_context, shap_feature = shap_result[0], shap_result[1]
            
            log("DeepSHAP explanation calculated successfully.", "INFO")
        except Exception as e:
             import traceback
             log(f"DeepSHAP calculation failed.\nError: {e}\nTraceback: {traceback.format_exc()}", "ERROR")
             shap_context = np.zeros(x.shape[1])
             shap_feature = np.zeros(x.shape[1])
        finally:
            self._is_explaining = False
            if hasattr(self, "context_net"):
                self.context_net._is_explaining = False

        # Return extended output
        return outputs + (shap_context, shap_feature)

    def seq2fix(self, x, pr_args={}):
        """
        Patched seq2fix that supports both LongTensor (Indices) and FloatTensor (Embeddings).
        """
        receptive_window, stride, out_channels = self.determinRF()

        # If input is embeddings (Float), we explicitly disable the internal embedding lookup
        # by temporarily mocking processRange or handling it here.
        # But processRange calls self.embd(x).
        # If x is Float, self.embd(x) will fail if self.embd is a real Embedding layer.
        # So we must Bypass embedding if x is Float.
        
        is_embedding_input = x.is_floating_point()
        
        # Padding
        if x.shape[1] < receptive_window:
            pad_len = receptive_window - x.shape[1]
            if is_embedding_input:
                # Pad with zeros in embedding space
                x = F.pad(x, (0, 0, 0, pad_len), value=0) # (B, L, D) -> pad last dim? No, dim 1 is Length.
                # F.pad for 3D tensor (B, L, D): (pad_left, pad_right, pad_top, pad_bottom...)
                # format is last dim backwards. D is last. We want to pad L (2nd to last).
                # (0, 0, 0, pad_len) -> Pad D by 0, Pad L by pad_len
            else:
                x = F.pad(x, (0, pad_len), value=0)

        batch_size = x.shape[0]
        length = x.shape[1]
        
        # Max-Pooling Logic (Chunking)
        # Optimization: Use cached winner indices if predicting for SHAP to ensure fixed graph and speed
        
        final_indices = None
        if self._is_explaining and getattr(self, "saved_indices", None) is not None:
             final_indices = self.saved_indices
             # log("Using saved winner indices for SHAP", "INFO")
             
             # Handle batch size mismatch (SHAP batches inputs)
             if len(final_indices) != batch_size:
                 # If we have saved indices (e.g. from Batch=1 prediction), 
                 # but SHAP passes multiple samples (e.g. baseline + input), 
                 # we broadcast the saved indices to ensure consistent graph structure for DeepSHAP.
                 if len(final_indices) == 1:
                     final_indices = [final_indices[0]] * batch_size
                 else:
                     # Fallback for unexpected batch sizes: try to broadcast index 0 or warn
                     # log(f"Warning: Saved indices size {len(final_indices)} != Batch size {batch_size}. Broadcasting index 0.", "WARNING")
                     final_indices = [final_indices[0]] * batch_size
        
        if final_indices is None:
            winner_values = np.zeros((batch_size, out_channels)) - 1.0
            winner_indices = np.zeros((batch_size, out_channels), dtype=np.int64)
                
            step = self.chunk_size 
            start = 0
            end = start+step
            
            with torch.no_grad():
                while start < end and (end-start) >= max(self.min_chunk_size, receptive_window):
                    x_sub = x[:,start:end]
                    
                    # Handling Embedding Input
                    if is_embedding_input:
                        # Bypass standard processRange which expects indices
                        x_emb = x_sub.transpose(1, 2) # (B, L, D) -> (B, D, L)
                        activs = self._process_embeddings(x_emb, gct=pr_args.get('gct'))
                    else:
                        activs = self.processRange(x_sub.long(), **pr_args)
                        
                    activ_win, activ_indx = F.max_pool1d(activs, kernel_size=activs.shape[2], return_indices=True)
                    
                    activ_win = activ_win.cpu().numpy()[:,:,0]
                    activ_indx = activ_indx.cpu().numpy()[:,:,0]
                    selected = winner_values < activ_win
                    winner_indices[selected] = activ_indx[selected]*stride + start 
                    winner_values[selected]  = activ_win[selected]
                    start = end
                    end = min(start+step, length)

            final_indices = [np.unique(winner_indices[b,:]) for b in range(batch_size)]
            
            # Save indices for SHAP use later
            if not self._is_explaining:
                 self.saved_indices = final_indices

        # Gathering Selected Chunks
        chunk_list = []
        for b in range(batch_size):
            chunks = []
            for i in final_indices[b]:
                s = max(i-receptive_window, 0)
                e = min(i+receptive_window, length)
                chunks.append(x[b:b+1, s:e])
            chunk_list.append(chunks)
            
        chunk_list = [torch.cat(c, dim=1)[0,:] for c in chunk_list]
        x_selected = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True)
        

        cur_device = next(self.parameters()).device
        if cur_device is not None:
            x_selected = x_selected.to(cur_device)

        # Final Processing of Selected Chunks (With Gradients)
        if is_embedding_input:
             x_emb = x_selected.transpose(1, 2)
             x_selected = self._process_embeddings(x_emb, gct=pr_args.get('gct'))
        else:
             x_selected = self.processRange(x_selected.long(), **pr_args)
             
        x_selected = self.pooling(x_selected)
        x_selected = x_selected.view(x_selected.size(0), -1)
            
        return x_selected

    def _process_embeddings(self, x, gct=None):
        """
        Helper to run convolutions on embeddings (skipping self.embd).
        x shape: (B, Channels/EmbSize, Length)
        """
        # Logic copied from MalConvGCT.processRange but starting after self.embd
        
        # x is already permuted to (B, C, L) by caller
        
        for conv_glu, linear_cntx, conv_share in zip(self.convs, self.linear_atn, self.convs_share):
            x = F.glu(conv_glu(x), dim=1)
            x = F.leaky_relu(conv_share(x))
            
            B = x.shape[0]
            C = x.shape[1]
            
            if gct is not None:
                ctnx = torch.tanh(linear_cntx(gct))
                ctnx = torch.unsqueeze(ctnx, dim=2)
                x_tmp = x.view(1,B*C,-1) 
                x_tmp = F.conv1d(x_tmp, ctnx, groups=B)
                x_gates = x_tmp.view(B, 1, -1)
                gates = torch.sigmoid( x_gates )
                x = x * gates
        
        return x

def compute_deep_shap(model, input_tensor, target_class=1):
    explainer = DeepShapExplainer(model)
    return explainer.explain(input_tensor, target_class)
