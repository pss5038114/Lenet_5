import torch
import os
from model import LeNet5

def export_weights():
    model = LeNet5()
    model.load_state_dict(torch.load("../lenet5_fp32.pth"))
    os.makedirs("weights_txt", exist_ok=True)

    for name, param in model.named_parameters():
        # 하드웨어에서 읽기 편하도록 1차원으로 펼쳐서 저장
        flat_data = param.detach().cpu().numpy().flatten()
        file_path = f"weights_txt/{name}.txt"
        
        with open(file_path, "w") as f:
            for val in flat_data:
                f.write(f"{val:.10f}\n") # 소수점 10자리까지 저장
        print(f"Exported: {name} ({len(flat_data)} values)")

if __name__ == "__main__":
    export_weights()