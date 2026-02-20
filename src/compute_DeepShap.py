import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import load_config, log
from MalConvGCT_nocat import MalConvGCT


class _DeepLiftPoolFunc(torch.autograd.Function):
    """AdaptiveMaxPool1d(1) + DeepLIFT Rescale Rule backward."""

    @staticmethod
    def forward(ctx, x, ref_x, ref_y):
        y, idx = F.adaptive_max_pool1d(x, 1, return_indices=True)
        ctx.save_for_backward(x, y, ref_x, ref_y, idx)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y, ref_x, ref_y, idx = ctx.saved_tensors

        config = load_config()
        eps = (
            config.get("explainability", {})
            .get("integrated_gradients", {})
            .get("epsilon", 1e-7)
        )

        delta_y = y - ref_y  # (B, C, 1)
        delta_x = x - ref_x  # (B, C, L)
        stable = delta_x.abs() > eps

        safe_dx = delta_x.clone()
        safe_dx[~stable] = 1.0
        scale = delta_y / safe_dx  # broadcast (B,C,1) / (B,C,L)
        scale[~stable] = 0.0

        grad_input = grad_output * scale

        # Fallback: δx ≈ 0 이면서 winner인 위치에 standard gradient 전달
        if (~stable).any():
            fallback = torch.zeros_like(x)
            fallback.scatter_(2, idx, grad_output)
            grad_input[~stable] = fallback[~stable]

        return grad_input, None, None


class _DeepLiftPool(nn.Module):
    def __init__(self, ref_x: torch.Tensor, ref_y: torch.Tensor):
        super().__init__()
        self.register_buffer("ref_x", ref_x)
        self.register_buffer("ref_y", ref_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DeepLiftPoolFunc.apply(x, self.ref_x, self.ref_y)


class IntegratedGradientsExplainer:
    def __init__(self, model: nn.Module):
        self.model = model
        self._load_config()

    def _load_config(self):
        config = load_config()
        ig_cfg = config.get("explainability", {}).get("integrated_gradients", {})

        self.n_steps = ig_cfg.get("n_steps", 50)
        self.target_class = ig_cfg.get("target_class", 1)
        self.baseline = ig_cfg.get("baseline", "zero")
        self.epsilon = ig_cfg.get("epsilon", 1e-7)

    # ── public API ──────────────────────────────────────────────────────

    def explain(self, input_tensor: torch.Tensor, target_class: int | None = None):
        """
        Per-byte [Context, Feature] attribution을 계산한다.

        Args:
            input_tensor: (1, L) LongTensor — 원본 바이트 시퀀스 (+1 인코딩).
            target_class: 설명 대상 클래스 인덱스. None이면 config 값 사용.

        Returns:
            [shap_ctx, shap_feat]: 각각 (L,) ndarray.
        """
        if target_class is None:
            target_class = self.target_class

        self.model.eval()
        original_low_mem = getattr(self.model, "low_mem", False)
        if hasattr(self.model, "low_mem"):
            self.model.low_mem = False

        try:
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                emb_main = self.model.embd(input_tensor)
                emb_ctx = self.model.context_net.embd(input_tensor)

            emb_combined = torch.cat([emb_ctx, emb_main], dim=-1)  # (B,L,2D)
            ref_combined = self._make_baseline(emb_combined)  # (B,L,2D)

            self.model._is_explaining = True
            self.model.context_net._is_explaining = True

            diff = emb_combined - ref_combined
            grad_acc = torch.zeros_like(emb_combined)

            for k in range(self.n_steps):
                alpha = (k + 0.5) / self.n_steps  # midpoint rule
                interp = (
                    (ref_combined + alpha * diff).detach().clone().requires_grad_(True)
                )

                self.model.zero_grad()
                logits = self.model(interp)
                logits[0, target_class].backward()

                grad_acc += interp.grad.detach()

            attr = ((grad_acc / self.n_steps) * diff).detach().cpu().numpy()

            D = emb_ctx.shape[-1]
            shap_ctx = attr[..., :D].sum(axis=-1)[0]  # (L,)
            shap_feat = attr[..., D:].sum(axis=-1)[0]  # (L,)

            return [shap_ctx, shap_feat]

        except Exception as e:
            import traceback

            log(f"IG Explain Error: {e}\n{traceback.format_exc()}", "ERROR")
            return [np.zeros(input_tensor.shape[1]), np.zeros(input_tensor.shape[1])]

        finally:
            if hasattr(self.model, "low_mem"):
                self.model.low_mem = original_low_mem
            self.model._is_explaining = False
            if hasattr(self.model, "context_net"):
                self.model.context_net._is_explaining = False

    def _make_baseline(self, emb_combined: torch.Tensor) -> torch.Tensor:
        if self.baseline == "zero":
            return torch.zeros_like(emb_combined)
        raise ValueError(f"Unknown baseline type: '{self.baseline}'")


class MalConvGCTExplainable(MalConvGCT):
    def __init__(self, **kwargs):
        config = load_config()
        model_cfg = config.get("model", {}).get("malconv", {})

        defaults = dict(
            out_size=model_cfg.get("num_classes", 2),
            channels=model_cfg.get("channels", 128),
            window_size=model_cfg.get("window_size", 512),
            stride=model_cfg.get("stride", 512),
            embd_size=model_cfg.get("embd_size", 8),
            layers=1,
            log_stride=None,
            low_mem=True,
        )
        defaults.update(kwargs)

        super().__init__(**defaults)
        self.embd_size = defaults["embd_size"]

        self.explainer = IntegratedGradientsExplainer(self)
        self._is_explaining = False

        import types

        self.seq2fix = types.MethodType(self.__class__.seq2fix, self)
        self._process_embeddings = types.MethodType(
            self.__class__._process_embeddings_gct, self
        )

        if hasattr(self, "context_net"):
            self.context_net.seq2fix = types.MethodType(
                self.__class__.seq2fix, self.context_net
            )
            self.context_net._process_embeddings = types.MethodType(
                self.__class__._process_embeddings_ml, self.context_net
            )
            self.context_net._is_explaining = False
            self.context_net.saved_indices = None

        for target in [self, getattr(self, "context_net", None)]:
            if (
                target
                and hasattr(target, "cat")
                and hasattr(target.cat, "_backward_hooks")
            ):
                target.cat._backward_hooks.clear()

    @staticmethod
    def _process_embeddings_gct(self, x, gct=None):
        """MalConvGCT processRange 로직 (embd 이후, Gating 포함)."""
        for conv_glu, linear_cntx, conv_share in zip(
            self.convs, self.linear_atn, self.convs_share
        ):
            x = F.leaky_relu(conv_share(F.glu(conv_glu(x), dim=1)))

            if gct is not None:
                B, C = x.shape[0], x.shape[1]
                ctnx = torch.tanh(linear_cntx(gct)).unsqueeze(2)
                x_tmp = F.conv1d(x.view(1, B * C, -1), ctnx, groups=B)
                gates = torch.sigmoid(x_tmp.view(B, 1, -1))
                x = x * gates
        return x

    @staticmethod
    def _process_embeddings_ml(self, x, gct=None):
        """MalConvML processRange 로직 (Gating 없음)."""
        for conv_glu, conv_share in zip(self.convs, self.convs_1):
            x = F.leaky_relu(conv_share(F.glu(conv_glu(x), dim=1)))
        return x

    # ── forward ─────────────────────────────────────────────────────

    def forward(self, x, *args):
        # Unpack SHAP-style args
        if args:
            x = [x, args[0]]

        # Case-3: IG 내부 루프에서 보내는 (B,L,2D) 결합 임베딩
        if (
            isinstance(x, torch.Tensor)
            and x.ndim == 3
            and x.shape[-1] == 2 * self.embd_size
        ):
            return self._forward_combined_embedding(x)

        # FloatTensor (적대적 공격 등) — 설명 없이 추론만
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return super().forward(x)

        # List/Tuple 임베딩 입력
        if isinstance(x, (list, tuple)):
            return self._forward_combined_embedding_pair(x[0], x[1])

        # 정수 입력 — 추론 중이거나 grad 비활성이면 설명 스킵
        if self._is_explaining or not torch.is_grad_enabled():
            return super().forward(x)

        # 일반 정수 입력 → 추론 + IG 설명
        return self._forward_with_explanation(x)

    def _forward_combined_embedding(self, x: torch.Tensor):
        emb_ctx = x[..., : self.embd_size]
        emb_main = x[..., self.embd_size :]

        global_context = self.context_net.seq2fix(emb_ctx)
        post_conv = self.seq2fix(emb_main, pr_args={"gct": global_context})

        return self.fc_2(F.leaky_relu(self.fc_1(post_conv)))

    def _forward_combined_embedding_pair(self, emb_ctx, emb_main):
        global_context = self.context_net.seq2fix(emb_ctx)
        post_conv = self.seq2fix(emb_main, pr_args={"gct": global_context})

        return self.fc_2(F.leaky_relu(self.fc_1(post_conv)))

    def _forward_with_explanation(self, x: torch.Tensor):
        outputs = super().forward(x)

        self._is_explaining = True
        if hasattr(self, "context_net"):
            self.context_net._is_explaining = True

        try:
            shap_ctx, shap_feat = self.explainer.explain(x)
            log("IG explanation calculated successfully.", "INFO")
        except Exception as e:
            import traceback

            log(f"IG explanation failed: {e}\n{traceback.format_exc()}", "ERROR")
            shap_ctx = np.zeros(x.shape[1])
            shap_feat = np.zeros(x.shape[1])
        finally:
            self._is_explaining = False
            if hasattr(self, "context_net"):
                self.context_net._is_explaining = False

        return outputs + (shap_ctx, shap_feat)

    def seq2fix(self, x, pr_args={}):
        receptive_window, stride, out_channels = self.determinRF()
        is_emb = x.is_floating_point()

        # Padding
        if x.shape[1] < receptive_window:
            pad = receptive_window - x.shape[1]
            x = F.pad(x, (0, 0, 0, pad) if is_emb else (0, pad), value=0)

        B, L = x.shape[0], x.shape[1]

        # ── Winner 인덱스 결정 (캐시 또는 새로 계산) ──
        final_indices = None

        if self._is_explaining and getattr(self, "saved_indices", None) is not None:
            final_indices = self.saved_indices
            if len(final_indices) != B:
                final_indices = [final_indices[0]] * B

        if final_indices is None:
            winner_vals = np.full((B, out_channels), -1.0)
            winner_idxs = np.zeros((B, out_channels), dtype=np.int64)

            step = self.chunk_size
            start = 0
            end = start + step

            with torch.no_grad():
                while start < end and (end - start) >= max(
                    self.min_chunk_size, receptive_window
                ):
                    sub = x[:, start:end]
                    if is_emb:
                        activs = self._process_embeddings(
                            sub.transpose(1, 2), gct=pr_args.get("gct")
                        )
                    else:
                        activs = self.processRange(sub.long(), **pr_args)

                    wins, idxs = F.max_pool1d(
                        activs, kernel_size=activs.shape[2], return_indices=True
                    )
                    wins = wins.cpu().numpy()[:, :, 0]
                    idxs = idxs.cpu().numpy()[:, :, 0]

                    sel = winner_vals < wins
                    winner_idxs[sel] = idxs[sel] * stride + start
                    winner_vals[sel] = wins[sel]

                    start = end
                    end = min(start + step, L)

            final_indices = [np.unique(winner_idxs[b, :]) for b in range(B)]

            if not self._is_explaining:
                self.saved_indices = final_indices

        # ── Winner 청크 수집 ──
        chunks = []
        for b in range(B):
            segs = [
                x[
                    b : b + 1,
                    max(i - receptive_window, 0) : min(i + receptive_window, L),
                ]
                for i in final_indices[b]
            ]
            chunks.append(torch.cat(segs, dim=1)[0, :])

        x_sel = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True)
        x_sel = x_sel.to(next(self.parameters()).device)

        if is_emb:
            x_sel = self._process_embeddings(
                x_sel.transpose(1, 2), gct=pr_args.get("gct")
            )
        else:
            x_sel = self.processRange(x_sel.long(), **pr_args)

        x_sel = self.pooling(x_sel)
        return x_sel.view(x_sel.size(0), -1)


def compute_deep_shap(model, input_tensor, target_class=None):
    explainer = IntegratedGradientsExplainer(model)
    return explainer.explain(input_tensor, target_class)
