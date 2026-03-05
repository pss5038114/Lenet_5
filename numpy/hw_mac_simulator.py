import numpy as np
import os

def symmetric_quantize(data, bits=8):
    # 대칭 양자화: 최대 절댓값을 기준으로 스케일링 (Zero-point = 0)
    max_val = np.max(np.abs(data))
    q_max = 2**(bits - 1) - 1 # 8비트면 127
    
    scale = max_val / q_max if max_val != 0 else 1e-8
    
    # Python에서 오버플로우를 막기 위해 임시로 int32 캐스팅 (값 자체는 -127 ~ 127 사이)
    q_data = np.round(data / scale).astype(np.int32) 
    q_data = np.clip(q_data, -q_max, q_max)
    return q_data, scale

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 원본 소수점(FP32) 데이터 로드 (test_image_0.txt와 weights_txt 폴더 사용)
    img_fp32 = np.loadtxt(os.path.join(BASE_DIR, "test_image_0.txt")).reshape(1, 32, 32)
    w_fp32 = np.loadtxt(os.path.join(BASE_DIR, "weights_txt", "conv1.weight.txt")).reshape(6, 1, 5, 5)
    b_fp32 = np.loadtxt(os.path.join(BASE_DIR, "weights_txt", "conv1.bias.txt"))
    
    # 2. 입력과 가중치를 대칭 양자화 (INT8 레벨 정수)
    img_int8, s_img = symmetric_quantize(img_fp32, 8)
    w_int8, s_w = symmetric_quantize(w_fp32, 8)
    
    # 3. Bias 양자화 (INT32)
    # [하드웨어 철칙]: Bias의 스케일은 반드시 (입력 스케일 * 가중치 스케일) 이어야 합니다.
    s_bias = s_img * s_w
    b_int32 = np.round(b_fp32 / s_bias).astype(np.int32)
    
    # 4. 하드웨어 MAC (Multiply-Accumulate) 시뮬레이션 시작
    out_c, in_c, k_h, k_w = w_int8.shape
    _, in_h, in_w = img_int8.shape
    out_h, out_w = in_h - k_h + 1, in_w - k_w + 1
    
    # 하드웨어의 32비트 레지스터(누적기) 배열
    out_int32 = np.zeros((out_c, out_h, out_w), dtype=np.int32)
    
    # 비교를 위한 FP32 골든 모델 결과 배열
    out_fp32_golden = np.zeros((out_c, out_h, out_w), dtype=np.float32)
    
    print("--- 칩 내부 MAC 연산 시뮬레이션 시작 ---")
    for oc in range(out_c):
        for ic in range(in_c):
            for i in range(out_h):
                for j in range(out_w):
                    # [진짜 하드웨어 동작 로직]
                    # -127~127 사이의 정수(img_int8)와 정수(w_int8)를 곱해서 더합니다.
                    # 소수점 계산은 단 1도 들어가지 않습니다!
                    mac_sum = np.sum(img_int8[ic, i:i+k_h, j:j+k_w] * w_int8[oc, ic])
                    out_int32[oc, i, j] += mac_sum
                    
                    # [FP32 골든 모델 동작 로직]
                    mac_sum_fp32 = np.sum(img_fp32[ic, i:i+k_h, j:j+k_w] * w_fp32[oc, ic])
                    out_fp32_golden[oc, i, j] += mac_sum_fp32
                    
        # 채널 누적이 끝나면 INT32 Bias를 덧셈기에 통과시킵니다.
        out_int32[oc] += b_int32[oc]
        out_fp32_golden[oc] += b_fp32[oc]
        
    # 5. 결과 검증 (역양자화)
    # 하드웨어에서 나온 32비트 정수 결과를 다시 원래 소수점으로 되돌려봅니다.
    out_dequantized = out_int32 * s_bias
    
    print("\n[ 결과 비교 (Conv1 첫 번째 채널, 1열 데이터 5개) ]")
    print(f"하드웨어 레지스터(INT32):  {out_int32[0, 0, :5]}")
    print(f"FP32 골든 모델 결과:       {np.round(out_fp32_golden[0, 0, :5], 4)}")
    print(f"INT32를 다시 소수점으로:   {np.round(out_dequantized[0, 0, :5], 4)}")
    
    error = np.abs(out_fp32_golden - out_dequantized).mean()
    print(f"\n평균 오차(Mean Absolute Error): {error:.6f}")
    if error < 0.05:
        print("🎉 대성공! 100% 정수로만 계산했는데 FP32 원본과 결과가 거의 똑같습니다!")

if __name__ == "__main__":
    main()