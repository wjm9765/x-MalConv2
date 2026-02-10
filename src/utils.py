import os
import yaml
import datetime

class Colors:
    RESET = "\033[0m"
    DEBUG = "\033[36m" # Cyan
    INFO = "\033[32m"  # Green
    WARNING = "\033[33m" # Yellow
    ERROR = "\033[31m" # Red

def log(message, level="INFO"):
    """
    Logs a message with timestamp and color based on severity.
    Usage: log("Model loaded", "INFO") or log("Shape mismatch", "DEBUG")
    """
    level = level.upper()
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    color = Colors.RESET
    if level == "DEBUG":
        color = Colors.DEBUG
    elif level == "INFO":
        color = Colors.INFO
    elif level == "WARNING":
        color = Colors.WARNING
    elif level == "ERROR":
        color = Colors.ERROR
        
    print(f"{Colors.RESET}[{timestamp}] {color}{level:<5}{Colors.RESET} : {message}")

def load_config(config_path=None):
    """
    config.yaml 파일을 로드하여 딕셔너리로 반환합니다.
    
    Args:
        config_path (str, optional): 설정 파일의 경로. 
                                     None일 경우 기본 경로(프로젝트 루트/config/config.yaml)를 탐색합니다.
    
    Returns:
        dict: 설정 값이 담긴 딕셔너리
    """
    if config_path is None:
        # 현재 파일(src/utils.py)의 상위 폴더(src)의 상위 폴더(Project Root)를 기준으로 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, 'config', 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config
