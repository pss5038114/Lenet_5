import torch
import os
import sys

# 현재 파일의 부모 폴더(Lenet_5)를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from model import LeNet5 # 이제 정상적으로 불러올 수 있습니다.

def export_weights():
    model = LeNet5()
    # 경로 수정: 상위 폴더에 있는 pth 파일을 읽어야 함
    model.load_state_dict(torch.load("../lenet5_fp32.pth"))
    os.makedirs("weights_txt", exist_ok=True)

    for name, param in model.named_parameters():
        flat_data = param.detach().cpu().numpy().flatten()
        file_path = f"weights_txt/{name}.txt"
        
        with open(file_path, "w") as f:
            for val in flat_data:
                f.write(f"{val:.10f}\n")
        print(f"Exported: {name} ({len(flat_data)} values)")

if __name__ == "__main__":
    export_weights()