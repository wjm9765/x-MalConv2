import os
import gzip
import numpy as np
import torch


# 수정할 것 max_len -> config에서 불러오도록
def preprocess_pe_file(file_path: str, mode: str = "default", max_len: int = 4000000):
    """
    EXE 파일을 MalConv2 모델 입력에 맞게 전처리하는 함수.
    BinaryLoader.py의 __getitem__ 로직을 기반으로 함.

    Args:
        file_path (str): 전처리할 PE 파일의 경로 (절대 경로 또는 상대 경로)
        mode (str): 처리 모드. 'default'는 일반 변환, 'attack'은 공격용 패딩 추가(구현 예정).
        max_len (int): 읽어들일 최대 바이트 수 (기본값: 4,000,000 바이트)

    Returns:
        torch.Tensor: 모델 입력용 텐서. Shape: (1, Sequence_Length)
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # 1. 파일 읽기 (gzip 압축된 경우 대응)
    try:
        with gzip.open(file_path, "rb") as f:
            raw_bytes = f.read(max_len)
    except OSError:
        # gzip이 아닌 일반 파일인 경우
        with open(file_path, "rb") as f:
            raw_bytes = f.read(max_len)

    # 2. 바이트 -> 정수 배열 변환
    # MalConv 모델은 0을 패딩 인덱스로 사용하므로, 바이트 값(0~255)에 1을 더해 1~256 범위로 변환
    x = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int16) + 1

    # 3. 모드에 따른 처리
    if mode == "default":
        # 기본 모드: 추가 처리 없음
        pass

    elif mode == "attack":
        # 공격 모드: 공격용 패딩 추가 로직 (추후 구현)
        # TODO: 여기에 적대적 예제 생성이나 패딩 추가 로직을 구현하세요.
        pass

    else:
        raise ValueError(
            f"지원하지 않는 모드입니다: {mode}. 'default' 또는 'attack'을 사용하세요."
        )

    # 4. 텐서 변환
    # 모델 입력 형태 (Batch_Size, Sequence_Length)에 맞추기 위해 차원 추가
    # Embedding 층 입력을 위해 LongTensor(int64)로 변환
    input_tensor = torch.tensor(x, dtype=torch.long).unsqueeze(0)

    return input_tensor
