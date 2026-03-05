import torch
from torchvision import datasets, transforms
import os

def export_test_image():
    # 1. 테스트 데이터셋 로드
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 상위 폴더의 data를 사용하도록 경로 지정
    dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    # 첫 번째 이미지(숫자 7)와 정답 가져오기
    img, label = dataset[0] 
    
    # 2. 하드웨어 메모리처럼 1D 텍스트로 저장
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "test_image_0.txt")
    
    with open(file_path, "w") as f:
        for val in img.flatten().numpy():
            f.write(f"{val:.6f}\n")
            
    print(f"Saved test image. True Label: {label} (Expected Output)")

if __name__ == "__main__":
    export_test_image()