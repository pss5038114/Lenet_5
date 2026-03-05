import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import LeNet5

def train():
    # 데이터셋 준비 (MNIST 28x28 -> 32x32로 리사이즈)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)

    model = LeNet5()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    print("학습 시작 (약 1~2분 소요)...")
    for epoch in range(2): # 빠른 확인을 위해 2번만 반복
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} 완료!")

    # 중요: FP32 가중치 저장
    torch.save(model.state_dict(), "lenet5_fp32.pth")
    print("학습 완료 및 'lenet5_fp32.pth' 저장 성공!")

if __name__ == "__main__":
    train()