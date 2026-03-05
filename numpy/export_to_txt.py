import torch
import os
import sys

# 1. 현재 스크립트 파일의 위치를 기준으로 상위 폴더(Lenet_5) 경로를 절대 경로로 구함
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(BASE_DIR)

from model import LeNet5

def export_weights():
    model = LeNet5()
    
    # 2. 가중치 파일 위치도 BASE_DIR을 이용해 정확한 절대 경로로 지정
    weight_path = os.path.join(BASE_DIR, "lenet5_fp32.pth")
    model.load_state_dict(torch.load(weight_path))
    
    # 3. 저장할 폴더 위치도 numpy 폴더 내부로 명확하게 지정
    save_dir = os.path.join(BASE_DIR, "numpy", "weights_txt")
    os.makedirs(save_dir, exist_ok=True)

    for name, param in model.named_parameters():
        flat_data = param.detach().cpu().numpy().flatten()
        file_path = os.path.join(save_dir, f"{name}.txt")
        
        with open(file_path, "w") as f:
            for val in flat_data:
                f.write(f"{val:.10f}\n")
        print(f"Exported: {name} ({len(flat_data)} values)")

if __name__ == "__main__":
    export_weights()