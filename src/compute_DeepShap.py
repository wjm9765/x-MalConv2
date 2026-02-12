import torch
import torch.nn.functional as F
import numpy as np
from .utils import load_config, log
from MalConvGCT_nocat import MalConvGCT


class DeepShapExplainer:
    """
    Implements DeepSHAP explanation for MalConvGCT models.

    This implementation adapts the Deep LIFT/SHAP algorithm for the specialized architecture
    of MalConvGCT (which includes Gated Convolutional Transformers and Max-Pooling over long sequences).

    Reference:
    - Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
    - Captum Library (PyTorch): https://captum.ai/

    Note for Open Source Users:
    - This implementation assumes 'target_class=1' (Malware) by default.
    - Low-memory optimizations in MalConvGCT are temporarily disabled during explanation to ensure correct gradient flow.
    """

    def __init__(self, model):
        self.model = model
        self.config = load_config()
        self.baseline_type = (
            self.config.get("explainability", {})
            .get("deep_shap", {})
            .get("baseline", "zero")
        )

        self.captured_embeddings = []

    def _get_baseline_input(self, input_tensor):
        """
        Generates baseline input (reference) for SHAP calculation.

        Default: Zero tensor (representing empty/padding bytes).

        [Customization Note]:
        If you want to use a different baseline (e.g., average embedding, or benign file sample),
        modify this method.
        """
        if self.baseline_type == "zero":
            return torch.zeros_like(input_tensor)
        else:
            # User can modify this to add other baseline types
            return torch.zeros_like(input_tensor)

    def explain(self, input_tensor, target_class=1):
        """
        Compute Deep SHAP values for Context and Feature separately.

        Args:
            input_tensor: (Batch, Length) integer tensor of byte values.
            target_class: The class index to explain (Default 1 for Malware).
        """
        self.model.eval()
        self.model.zero_grad()

        original_low_mem = getattr(self.model, "low_mem", False)
        if hasattr(self.model, "low_mem"):
            self.model.low_mem = False

        if hasattr(self.model, "context_net"):
            self.model.context_net._shap_captured_indices = []

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
        if hasattr(self.model, "context_net") and hasattr(
            self.model.context_net, "embd"
        ):
            if self.model.context_net.embd is not self.model.embd:
                h2 = self.model.context_net.embd.register_forward_hook(
                    embedding_forward_hook
                )
                handles.append(h2)

        shap_maps = []
        try:
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
            indices_list = []
            if hasattr(self.model, "context_net") and hasattr(
                self.model.context_net, "_shap_captured_indices"
            ):
                indices_list.extend(self.model.context_net._shap_captured_indices)
            indices_list.extend(self.model._shap_captured_indices)

            with torch.no_grad():
                ref_emb_vec = self.model.embd(torch.tensor([0]).to(input_tensor.device))

            for i in range(len(embeddings_list)):
                if i >= len(indices_list):
                    break

                emb_out = embeddings_list[i]
                indices_batch = indices_list[i]

                if emb_out.grad is None:
                    # Append zero map if no grad
                    shap_maps.append(np.zeros(total_len))
                    continue

                grad = emb_out.grad

                # DeepLIFT/DeepSHAP Formula: (Gradient * (Input - Reference))
                diff = emb_out - ref_emb_vec
                shap_per_token = (grad * diff).sum(dim=-1)  # (B, L_chunk)

                shap_values_np = shap_per_token.detach().cpu().numpy()[0]

                final_shap_map = np.zeros(total_len)
                real_indices = indices_batch[0]

                current_ptr = 0

                if i == 0 and hasattr(self.model, "context_net"):
                    rf, _, _ = self.model.context_net.determinRF()
                else:
                    rf, _, _ = self.model.determinRF()

                for idx_counter, start_idx in enumerate(real_indices):
                    s = max(start_idx - rf, 0)
                    e = min(start_idx + rf, total_len)
                    if e > total_len:
                        e = total_len
                    chunk_len = e - s

                    if current_ptr + chunk_len <= len(shap_values_np):
                        shap_chunk = shap_values_np[
                            current_ptr : current_ptr + chunk_len
                        ]
                        final_shap_map[s:e] += shap_chunk
                        current_ptr += chunk_len
                    else:
                        # If size mismatch (e.g. padding/chunks), break or fill
                        # Usually exact match if logic is correct
                        log(
                            f"Layer {i}: Size mismatch! Need {chunk_len}, have {len(shap_values_np) - current_ptr}",
                            "WARNING",
                        )

                shap_maps.append(final_shap_map)

        except Exception as e:
            log(f"Exception during DeepSHAP explanation: {str(e)}", "ERROR")

        finally:
            for h in handles:
                h.remove()
            if hasattr(self.model, "low_mem"):
                self.model.low_mem = original_low_mem
            if hasattr(self.model, "_shap_captured_indices"):
                del self.model._shap_captured_indices
            if hasattr(self.model, "context_net") and hasattr(
                self.model.context_net, "_shap_captured_indices"
            ):
                del self.model.context_net._shap_captured_indices

        return shap_maps


class MalConvGCTDeepShap(MalConvGCT):
    """
    Wrapper class for MalConvGCT to enable DeepSHAP explanation.

    This class patches the `seq2fix` method to capture the max-pooling indices
    during the forward pass, which are necessary to map the importance values
    back to the original byte locations.
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
        self.explainer = DeepShapExplainer(self)
        self._is_explaining = False
        if hasattr(self, "context_net"):
            import types

            self.context_net.seq2fix = types.MethodType(
                self.__class__.seq2fix, self.context_net
            )
        # --------------------------------

    def forward(self, x):
        if self._is_explaining:
            if hasattr(self, "context_net"):
                self.context_net._is_explaining = True
            return super().forward(x)

        outputs = super().forward(x)

        # Compute SHAP
        self._is_explaining = True
        try:
            # We assume we want to explain the prediction of the current forward pass
            target = 1

            shap_maps = self.explainer.explain(x, target_class=target)

            if len(shap_maps) >= 2:
                shap_context = shap_maps[0]
                shap_feature = shap_maps[1]
                log(
                    "DeepSHAP explanation computed for both Context and Feature.",
                    "INFO",
                )
            elif len(shap_maps) == 1:
                shap_context = shap_maps[0]
                shap_feature = np.zeros_like(shap_maps[0])
                log(
                    "Only one SHAP map computed, setting feature SHAP to zero.",
                    "WARNING",
                )
            else:
                shap_context = np.zeros(x.shape[1])
                shap_feature = np.zeros(x.shape[1])
                log("No SHAP maps computed, setting both to zero.", "WARNING")

        finally:
            self._is_explaining = False
            if hasattr(self, "context_net"):
                self.context_net._is_explaining = False

        # Return extended output
        return outputs + (shap_context, shap_feature)

    def seq2fix(self, x, pr_args={}):
        """
        Overridden seq2fix from LowMemConvBase to capture indices for DeepSHAP.
        """
        receptive_window, stride, out_channels = self.determinRF()

        if x.shape[1] < receptive_window:
            x = F.pad(x, (0, receptive_window - x.shape[1]), value=0)

        batch_size = x.shape[0]
        length = x.shape[1]

        winner_values = np.zeros((batch_size, out_channels)) - 1.0
        winner_indices = np.zeros((batch_size, out_channels), dtype=np.int64)

        if not hasattr(self, "device_ids"):
            cur_device = next(self.embd.parameters()).device
        else:
            cur_device = None

        step = self.chunk_size
        start = 0
        end = start + step

        with torch.no_grad():
            while start < end and (end - start) >= max(
                self.min_chunk_size, receptive_window
            ):
                x_sub = x[:, start:end]
                if cur_device is not None:
                    x_sub = x_sub.to(cur_device)
                activs = self.processRange(x_sub.long(), **pr_args)
                activ_win, activ_indx = F.max_pool1d(
                    activs, kernel_size=activs.shape[2], return_indices=True
                )

                activ_win = activ_win.cpu().numpy()[:, :, 0]
                activ_indx = activ_indx.cpu().numpy()[:, :, 0]
                selected = winner_values < activ_win
                winner_indices[selected] = activ_indx[selected] * stride + start
                winner_values[selected] = activ_win[selected]
                start = end
                end = min(start + step, length)

        final_indices = [np.unique(winner_indices[b, :]) for b in range(batch_size)]

        is_explaining = getattr(self, "_is_explaining", False)

        if is_explaining:
            if not hasattr(self, "_shap_captured_indices"):
                self._shap_captured_indices = []
            self._shap_captured_indices.append(final_indices)

        chunk_list = [
            [
                x[
                    b : b + 1,
                    max(i - receptive_window, 0) : min(i + receptive_window, length),
                ]
                for i in final_indices[b]
            ]
            for b in range(batch_size)
        ]
        chunk_list = [torch.cat(c, dim=1)[0, :] for c in chunk_list]

        x_selected = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True)

        if cur_device is not None:
            x_selected = x_selected.to(cur_device)
        x_selected = self.processRange(x_selected.long(), **pr_args)
        x_selected = self.pooling(x_selected)
        x_selected = x_selected.view(x_selected.size(0), -1)

        return x_selected


def compute_deep_shap(model, input_tensor, target_class=1):
    explainer = DeepShapExplainer(model)
    return explainer.explain(input_tensor, target_class)
