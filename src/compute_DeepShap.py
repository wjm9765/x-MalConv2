import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils import load_config
from contextlib import contextmanager

# models 패키지 경로를 sys.path에 추가하는 것은 외부(test.ipynb 등)에서 수행했다고 가정
# 또는 필요시 여기서 추가. 여기서는 MalConv2-main 모델의 클래스 구조를 참고하기 위해
# 동적으로 import 하거나, 인자로 모델 객체를 받는 방식을 사용.

class DeepShapExplainer:
    def __init__(self, model):
        self.model = model
        self.config = load_config()
        self.baseline_type = self.config['explainability']['deep_shap']['baseline']
        self.approximation = self.config['explainability']['deep_shap']['approximation']
        
        # 데이터를 저장할 공간
        self.ref_activations = {}
        self.actual_activations = {}
        self.captured_indices = [] # selected chunks indices
        
        # Monkey Patch를 위한 원본 메서드 저장
        from models.MalConv2_main.LowMemConv import LowMemConvBase
        self.original_seq2fix = LowMemConvBase.seq2fix

    def _get_baseline_input(self, input_tensor):
        """
        기준값(Baseline) 생성
        MalConv의 경우 0 (Padding)을 기준으로 함.
        """
        if self.baseline_type == 'zero':
            return torch.zeros_like(input_tensor)
        elif self.baseline_type == 'embedding_mean':
            # 임베딩 평균은 구현 복잡도가 높으므로 일단 0으로 처리하거나 추후 구현
            return torch.zeros_like(input_tensor) 
        else:
            return torch.zeros_like(input_tensor)

    def _custom_seq2fix(self_instance, x, pr_args={}):
        """
        LowMemConvBase.seq2fix 메서드를 대체(Monkey Patch)할 함수.
        목적: LowMemConv가 내부적으로 선택(max-pooling)한 청크의 인덱스와 
              임베딩된 입력을 캡처하여 역전파 시 사용하기 위함.
        """
        # LowMemConv.py의 원본 로직을 그대로 가져오되, 필요한 정보를 self_instance._shap_data 에 저장
        
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
        
        # --- [SHAP Capture Logic] ---
        # 캡처된 정보를 인스턴스에 저장 (멀티 스레드 환경 아님을 가정)
        # context_net과 main net이 모두 이 함수를 쓰므로, 호출 순서대로 저장됨
        if not hasattr(self_instance, '_shap_captured_indices'):
            self_instance._shap_captured_indices = []
            
        self_instance._shap_captured_indices.append(final_indices)
        
        # ----------------------------

        if cur_device is not None:
            x_selected = x_selected.to(cur_device)
            
        # 여기서 Embedding 레이어의 출력을 캡처해야 함. (Gradient의 시작점)
        # processRange 내부에서 embd(x)를 호출함.
        # 따라서 우리는 processRange를 호출하되, 그 입력인 x_selected 자체에 대한 Gradient를 구할 수는 없음 (LongTensor).
        # Embedding Layer에 Hook을 걸어서 해결해야 함.
        
        output = self_instance.processRange(x_selected.long(), **pr_args)
        
        # 저장된 정보를 바탕으로 나중에 매핑하기 위해 x_selected 자체는 저장하지 않아도 됨.
        # Embedding Layer Hook에서 x_selected에 해당하는 임베딩 벡터를 잡을 것임.
        
        output = self_instance.pooling(output)
        output = output.view(output.size(0), -1)
            
        return output

    @contextmanager
    def _patch_model(self):
        """LowMemConvBase.seq2fix를 안전하게 패치하고 복구하는 컨텍스트 매니저"""
        from models.MalConv2_main.LowMemConv import LowMemConvBase
        original = LowMemConvBase.seq2fix
        LowMemConvBase.seq2fix = self._custom_seq2fix
        try:
            yield
        finally:
            LowMemConvBase.seq2fix = original

    def explain(self, input_tensor, target_class=1):
        """
        Deep SHAP 값을 계산하여 반환
        
        Args:
            input_tensor: (1, L) 형태의 입력 LongTensor
            target_class: 설명하고자 하는 클래스 (0: 정상, 1: 악성)
            
        Returns:
            shap_values: 입력과 동일한 길이의 1차원 SHAP 값 (numpy array)
        """
        self.model.eval()
        self.model.zero_grad()
        
        # 기준값 생성 (Zero Padding)
        ref_input = self._get_baseline_input(input_tensor).to(input_tensor.device)
        input_tensor = input_tensor.to(input_tensor.device) # Ensure device

        # 데이터 저장소 초기화
        self.model._shap_captured_indices = []
        self.captured_embeddings = [] # 임베딩 벡터와 그라디언트를 저장할 리스트

        # Embedding Layer Hook 정의
        def embedding_forward_hook(module, input, output):
            # output: (B, C, L) or (B, L, C) depending on implementation
            # MalConv code: x = self.embd(x); x = x.permute(0,2,1) in processRange
            # Hook output is (B, L, Embd_Size).
            # We need to retain grad for this output.
            if output.requires_grad:
                output.retain_grad()
            self.captured_embeddings.append(output)

        # Hook 등록
        handle = self.model.embd.register_forward_hook(embedding_forward_hook)

        try:
            with self._patch_model():
                # Actual Forward & Backward with Approximation
                outputs = self.model(input_tensor)
                logits = outputs[0]
                
                # Target Class에 대한 Score
                target_score = logits[0, target_class]
                
                # Backward Pass
                self.model.zero_grad()
                target_score.backward()
                
                # SHAP 값 계산 Logic
                total_len = input_tensor.shape[1]
                final_shap_map = np.zeros(total_len)
                
                embeddings_list = self.captured_embeddings
                indices_list = self.model._shap_captured_indices
                
                # Embedding(0) 값 구하기 (Reference)
                with torch.no_grad():
                    ref_emb_vec = self.model.embd(torch.tensor([0]).to(input_tensor.device))
                    # shape: (1, embd_size)
                
                # captured_embeddings에는 context_net과 main_net의 임베딩이 모두 들어옴
                # indices_list도 마찬가지
                # MalConvGCT 구조상: context_net -> main_net 순서로 호출됨
                # 따라서 zip으로 묶어서 처리 가능
                
                for i, (emb_out, indices_batch) in enumerate(zip(embeddings_list, indices_list)):
                    if emb_out.grad is None:
                        continue
                        
                    grad = emb_out.grad
                    
                    # 수식: phi = grad * (x - E[x])
                    # x: emb_out (Actual Embedding)
                    # E[x]: ref_emb_vec (Zero Padding Embedding)
                    
                    diff = emb_out - ref_emb_vec
                    
                    # Element-wise multiplication followed by sum over embedding dimension
                    shap_per_token = (grad * diff).sum(dim=-1) # (B, L)
                    
                    shap_values_np = shap_per_token.detach().cpu().numpy()[0] # Batch 0
                    
                    real_indices = indices_batch[0] 
                    
                    current_ptr = 0
                    receptive_window, _, _ = self.model.determinRF()
                    
                    for start_idx in real_indices:
                        s = max(start_idx - receptive_window, 0)
                        e = min(start_idx + receptive_window, total_len)
                        if e > total_len: e = total_len
                        
                        chunk_len = e - s
                        
                        if current_ptr + chunk_len <= len(shap_values_np):
                            shap_chunk = shap_values_np[current_ptr : current_ptr + chunk_len]
                            final_shap_map[s:e] += shap_chunk
                            current_ptr += chunk_len
                        else:
                            break

                return final_shap_map

        finally:
            handle.remove()

def compute_deep_shap(model, input_tensor, target_class=1):
    """
    외부에서 호출 가능한 래퍼 함수
    """
    explainer = DeepShapExplainer(model)
    return explainer.explain(input_tensor, target_class)
