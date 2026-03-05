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
    # 1. 현재 스크립트(quantize_to_txt.py)가 있는 'numpy' 폴더의 절대 경로를 구합니다.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 절대 경로를 바탕으로 폴더 위치를 명확하게 지정합니다.
    input_dir = os.path.join(BASE_DIR, "weights_txt")
    output_dir = os.path.join(BASE_DIR, "weights_int8_txt")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 이제 절대 경로(input_dir)에서 파일 목록을 가져옵니다.
    files = [f for f in os.listdir(input_dir) if "weight" in f]
    
    print("--- INT8 Quantization Table ---")
    for f in files:
        # 파일 경로도 정확하게 합쳐서 넘겨줍니다.
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f)
        
        s, z = quantize_and_save_int8(in_path, out_path)
        print(f"File: {f:20} | Scale: {s:.6f} | ZP: {z}")

if __name__ == "__main__":
    main()