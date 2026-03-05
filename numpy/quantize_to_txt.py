import numpy as np
import os

def quantize_and_save_int8(input_file, output_file):
    data = np.loadtxt(input_file)
    
    # 가중치 범위 기반 Scale/ZP 계산
    fp_min, fp_max = data.min(), data.max()
    q_min, q_max = -128, 127
    
    scale = (fp_max - fp_min) / (q_max - q_min)
    if scale == 0: scale = 1e-8
    zp = np.round(q_min - (fp_min / scale)).astype(int)
    
    # 양자화 (정수 변환)
    q_data = np.round(data / scale + zp).astype(int)
    q_data = np.clip(q_data, q_min, q_max)
    
    # 하드웨어에서 읽기 좋게 정수형태로 저장
    with open(output_file, "w") as f:
        for val in q_data:
            f.write(f"{val}\n")
            
    return scale, zp

def main():
    os.makedirs("weights_int8_txt", exist_ok=True)
    input_dir = "weights_txt"
    files = [f for f in os.listdir(input_dir) if "weight" in f]
    
    print("--- INT8 Quantization Table ---")
    for f in files:
        s, z = quantize_and_save_int8(f"{input_dir}/{f}", f"weights_int8_txt/{f}")
        print(f"File: {f:20} | Scale: {s:.6f} | ZP: {z}")

if __name__ == "__main__":
    main()