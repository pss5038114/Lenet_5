import torch
import numpy as np
from model import LeNet5

def quantize_tensor(tensor, num_bits=8):
    q_min, q_max = -2**(num_bits - 1), 2**(num_bits - 1) - 1
    
    fp_min = tensor.min().item()
    fp_max = tensor.max().item()
    
    # Scale 계산 (0으로 나누기 방지)
    scale = (fp_max - fp_min) / (q_max - q_min)
    if scale == 0: scale = 1e-8
    
    # Zero-point 계산
    zero_point = round(q_min - (fp_min / scale))
    zero_point = max(q_min, min(q_max, zero_point)) # 범위 제한
    
    # 양자화 및 클램핑
    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor = torch.clamp(q_tensor, q_min, q_max)
    
    # 다시 FP32로 역양자화 (추론 시뮬레이션을 위해)
    dq_tensor = scale * (q_tensor - zero_point)
    
    return dq_tensor, scale, zero_point

def main():
    # 1. 원본 가중치 로드
    model = LeNet5()
    model.load_state_dict(torch.load("lenet5_fp32.pth"))
    state_dict = model.state_dict()
    
    quantized_state_dict = {}
    
    print("가중치 양자화 진행 중...")
    for name, param in state_dict.items():
        if 'weight' in name:
            dq_param, s, z = quantize_tensor(param)
            quantized_state_dict[name] = dq_param
            print(f"Layer: {name:20} | Scale: {s:.6f} | ZP: {z}")
        else:
            # Bias(편향)는 보통 양자화하지 않거나 더 높은 비트를 쓰지만, 
            # 여기서는 그대로 유지하거나 똑같이 처리할 수 있습니다.
            quantized_state_dict[name] = param

    # 2. 8비트 가중치가 적용된 모델 저장
    torch.save(quantized_state_dict, "lenet5_int8_simulated.pth")
    print("\n8비트 양자화 모델(시뮬레이션 버전) 저장 완료!")

if __name__ == "__main__":
    main()