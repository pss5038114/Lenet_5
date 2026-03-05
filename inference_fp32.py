import torch
import time
from torchvision import datasets, transforms
from model import LeNet5

def run_inference():
    # 1. 모델 설정 및 가중치 로드
    model = LeNet5()
    model.load_state_dict(torch.load("lenet5_fp32.pth"))
    model.eval() # 추론 모드 (중요!)

    # 2. 테스트 데이터셋 준비
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    correct = 0
    total = len(test_loader)
    
    print(f"FP32 모델 추론 시작 (총 {total}장)...")
    
    start_time = time.time()
    with torch.no_grad(): # 기울기 계산 비활성화 (메모리 절약 및 속도 향상)
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    end_time = time.time()

    # 3. 결과 출력
    accuracy = 100. * correct / total
    duration = end_time - start_time
    
    print("-" * 30)
    print(f"결과 리포트 (FP32)")
    print(f"정확도: {accuracy:.2f}%")
    print(f"전체 추론 시간: {duration:.4f} 초")
    print(f"장당 평균 시간: {(duration/total)*1000:.4f} ms")
    print("-" * 30)

if __name__ == "__main__":
    run_inference()