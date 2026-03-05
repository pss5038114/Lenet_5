import numpy as np
import os

class LeNet5HardwareSim:
    def __init__(self, weight_dir="weights_txt"):
        self.w = {}
        self.s_w = {}
        self.b_fp32 = {}
        self.weight_dir = weight_dir
        
        # 모델의 모든 가중치를 대칭 양자화(Symmetric Quantize)하여 로드
        self._load_and_quantize("conv1", (6, 1, 5, 5), (6,))
        self._load_and_quantize("conv2", (16, 6, 5, 5), (16,))
        self._load_and_quantize("fc1", (120, 400), (120,))
        self._load_and_quantize("fc2", (84, 120), (84,))
        self._load_and_quantize("fc3", (10, 84), (10,))

    def _sym_quantize(self, data, bits=8):
        # 대칭 양자화: 영점(Zero-point) 없이 스케일만 계산
        max_val = np.max(np.abs(data))
        q_max = 2**(bits - 1) - 1
        scale = max_val / q_max if max_val != 0 else 1e-8
        q_data = np.round(data / scale).astype(np.int32)
        return np.clip(q_data, -q_max, q_max), scale
        
    def _load_and_quantize(self, name, w_shape, b_shape):
        w_data = np.loadtxt(f"{self.weight_dir}/{name}.weight.txt").reshape(w_shape)
        b_data = np.loadtxt(f"{self.weight_dir}/{name}.bias.txt").reshape(b_shape)
        
        self.w[name], self.s_w[name] = self._sym_quantize(w_data, 8)
        # Bias는 입력(Input) 스케일이 들어와야 정확한 INT32로 바꿀 수 있으므로 원본 유지
        self.b_fp32[name] = b_data

    def hw_conv2d(self, x_int8, s_in, name):
        w_int8 = self.w[name]
        s_w = self.s_w[name]
        
        # 1. Bias 양자화 (INT32) : 스케일 = 입력 스케일 * 가중치 스케일
        s_mac = s_in * s_w
        b_int32 = np.round(self.b_fp32[name] / s_mac).astype(np.int32)
        
        # 2. MAC 곱셈 및 누적 (100% 정수 연산)
        out_c, in_c, k_h, k_w = w_int8.shape
        _, in_h, in_w = x_int8.shape
        out_h, out_w = in_h - k_h + 1, in_w - k_w + 1
        out_int32 = np.zeros((out_c, out_h, out_w), dtype=np.int32)
        
        for oc in range(out_c):
            for ic in range(in_c):
                for i in range(out_h):
                    for j in range(out_w):
                        out_int32[oc, i, j] += np.sum(x_int8[ic, i:i+k_h, j:j+k_w] * w_int8[oc, ic])
            out_int32[oc] += b_int32[oc]
            
        return out_int32, s_mac

    def hw_linear(self, x_int8, s_in, name):
        w_int8 = self.w[name]
        s_w = self.s_w[name]
        
        s_mac = s_in * s_w
        b_int32 = np.round(self.b_fp32[name] / s_mac).astype(np.int32)
        
        # MAC (100% 정수 연산)
        out_int32 = np.zeros(w_int8.shape[0], dtype=np.int32)
        for oc in range(w_int8.shape[0]):
            out_int32[oc] = np.sum(x_int8 * w_int8[oc]) + b_int32[oc]
            
        return out_int32, s_mac

    def hw_activation_and_requantize(self, mac_int32, s_mac):
        # 칩 내부에서는 Look-Up Table(표)을 쓰지만, 시뮬레이터이므로 수식으로 대체
        # INT32 값을 잠시 복원 -> Tanh 적용 -> 다시 INT8로 재양자화
        fp32_val = np.tanh(mac_int32 * s_mac)
        out_int8, s_out = self._sym_quantize(fp32_val, 8)
        return out_int8, s_out

    def hw_avg_pool(self, x_int8, s_in, kernel=2, stride=2):
        # 하드웨어 덧셈기(Adder)와 시프트(Shift) 연산 모사
        C, H, W = x_int8.shape
        out_h, out_w = H // stride, W // stride
        out_int8 = np.zeros((C, out_h, out_w), dtype=np.int32)
        
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    # 정수 4개를 더하고 4로 나눔 (비트 시프트 연산과 동일)
                    pool_sum = np.sum(x_int8[c, i*stride:i*stride+kernel, j*stride:j*stride+kernel])
                    out_int8[c, i, j] = np.round(pool_sum / (kernel * kernel))
                    
        return np.clip(out_int8, -127, 127).astype(np.int32), s_in

    def forward(self, img_int8, s_img):
        print("--- 하드웨어 내부 데이터 흐름 ---")
        # Conv1 -> Tanh
        x, s_mac = self.hw_conv2d(img_int8, s_img, "conv1")
        x, s_x = self.hw_activation_and_requantize(x, s_mac)
        print(f"C1 통과  : {x.shape} (INT8)")
        
        # Pool1
        x, s_x = self.hw_avg_pool(x, s_x)
        print(f"S2 통과  : {x.shape} (INT8)")
        
        # Conv2 -> Tanh
        x, s_mac = self.hw_conv2d(x, s_x, "conv2")
        x, s_x = self.hw_activation_and_requantize(x, s_mac)
        print(f"C3 통과  : {x.shape} (INT8)")
        
        # Pool2
        x, s_x = self.hw_avg_pool(x, s_x)
        print(f"S4 통과  : {x.shape} (INT8)")
        
        # Flatten
        x = x.flatten()
        
        # FC1 -> Tanh
        x, s_mac = self.hw_linear(x, s_x, "fc1")
        x, s_x = self.hw_activation_and_requantize(x, s_mac)
        print(f"FC1 통과 : {x.shape} (INT8)")
        
        # FC2 -> Tanh
        x, s_mac = self.hw_linear(x, s_x, "fc2")
        x, s_x = self.hw_activation_and_requantize(x, s_mac)
        print(f"FC2 통과 : {x.shape} (INT8)")
        
        # FC3 (출력층은 활성화 함수 없이 32비트 결과 그대로 방출)
        x, s_mac = self.hw_linear(x, s_x, "fc3")
        final_out_fp32 = x * s_mac # 최종 확률 비교를 위해 복원
        print(f"출력단   : {final_out_fp32.shape} (INT32 -> FP32 변환)")
        
        return final_out_fp32

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 원본 이미지 로드 및 대칭 양자화 (ZP=0)
    img_fp32 = np.loadtxt(os.path.join(BASE_DIR, "test_image_0.txt")).reshape(1, 32, 32)
    max_val = np.max(np.abs(img_fp32))
    s_img = max_val / 127.0
    img_int8 = np.round(img_fp32 / s_img).astype(np.int32)
    
    # 2. 하드웨어 시뮬레이터 초기화 및 추론
    model = LeNet5HardwareSim(weight_dir=os.path.join(BASE_DIR, "weights_txt"))
    out = model.forward(img_int8, s_img)
    
    # 3. 결과 확인
    pred = np.argmax(out)
    print("\n==============================")
    print("[ 최종 칩(Chip) 시뮬레이션 결과 ]")
    print(f"-> 칩이 예측한 숫자 : {pred}")
    print(f"-> 실제 정답 숫자   : 7")
    print("==============================")

if __name__ == "__main__":
    main()