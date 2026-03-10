import numpy as np
import os
import time
import sys
from torchvision import datasets, transforms
from hw_full_int8_inference import LeNet5HardwareSim

# 1만 장 추론 시 터미널에 print 문이 너무 많이 찍히는 것을 방지하기 위한 유틸리티
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def evaluate_hardware_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 테스트 데이터 10,000장 로드
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 상위 폴더의 데이터셋 사용
    dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    # 2. 하드웨어 시뮬레이터 모델 초기화
    print("칩(Chip) 시뮬레이터 모델(INT8/INT32) 로딩 중...")
    model = LeNet5HardwareSim(weight_dir=os.path.join(BASE_DIR, "weights_txt"))
    
    correct = 0
    total = len(dataset)
    
    print(f"\n총 {total}장 하드웨어 시뮬레이션 추론 시작...")
    print("(주의: 파이썬(NumPy)으로 하드웨어 연산을 100% 모사하므로 CPU에 따라 1~2분 정도 소요될 수 있습니다.)\n")
    
    start_time = time.time()
    
    # 데이터셋 10,000장에 대해 반복
    for i in range(total):
        img_tensor, label = dataset[i]
        img_fp32 = img_tensor.numpy()
        
        # 이미지 대칭 양자화 로직 (ZP=0)
        max_val = np.max(np.abs(img_fp32))
        s_img = max_val / 127.0 if max_val != 0 else 1e-8
        img_int8 = np.round(img_fp32 / s_img).astype(np.int32)
        
        # 추론 과정의 print 출력 숨기기
        with HiddenPrints():
            out = model.forward(img_int8, s_img)
            
        pred = np.argmax(out)
        
        if pred == label:
            correct += 1
            
        # 1000장마다 진행 상황 출력
        if (i + 1) % 1000 == 0:
            print(f"[{i+1:5d} / {total}] 진행 중... 현재까지의 정확도: {100 * correct / (i + 1):.2f}%")
            
    end_time = time.time()
    
    print("\n" + "="*45)
    print("[ 최종 NPU 하드웨어 모델(Golden Model) 평가 결과 ]")
    print(f"-> 베이스라인(FP32) 정확도 : 약 98.28%")
    print(f"-> 하드웨어(INT8) 정확도   : {100 * correct / total:.2f}%")
    print(f"-> 1만 장 총 소요 시간     : {end_time - start_time:.2f} 초")
    print("="*45)

if __name__ == "__main__":
    evaluate_hardware_model()