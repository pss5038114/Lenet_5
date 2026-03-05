import numpy as np
import os

def quantize_test_image():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(BASE_DIR, "test_image_0.txt")
    output_file = os.path.join(BASE_DIR, "test_image_0_int8.txt")
    
    # 1. FP32 이미지 로드
    img = np.loadtxt(input_file)
    
    # 2. 이미지의 Scale과 Zero-point 계산
    fp_min, fp_max = img.min(), img.max()
    q_min, q_max = -128, 127
    
    scale = (fp_max - fp_min) / (q_max - q_min)
    if scale == 0: scale = 1e-8
    zp = int(np.round(q_min - (fp_min / scale)))
    
    # 3. 양자화 (INT8)
    q_img = np.round(img / scale + zp).astype(int)
    q_img = np.clip(q_img, q_min, q_max)
    
    # 4. 하드웨어 메모리용 텍스트 저장
    with open(output_file, "w") as f:
        for val in q_img:
            f.write(f"{val}\n")
            
    print(f"--- Image Quantization ---")
    print(f"Image Scale: {scale:.8f} | ZP: {zp}")
    print(f"Saved: test_image_0_int8.txt")

if __name__ == "__main__":
    quantize_test_image()