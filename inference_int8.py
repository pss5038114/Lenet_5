import torch
from torchvision import datasets, transforms
from model import LeNet5
import time

def run_inference():
    model = LeNet5()
    # 8비트로 '깎인' 가중치 로드
    model.load_state_dict(torch.load("lenet5_int8_simulated.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform), batch_size=1)

    correct = 0
    start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    accuracy = 100. * correct / len(test_loader)
    print(f"INT8 양자화 모델 정확도: {accuracy:.2f}%")
    print(f"소요 시간: {time.time() - start_time:.4f}초")

if __name__ == "__main__":
    run_inference()