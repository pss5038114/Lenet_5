import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageDraw

# 우리가 방금 만든 INT8 하드웨어 시뮬레이터 모듈을 불러옵니다.
from hw_full_int8_inference import LeNet5HardwareSim

class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NPU 하드웨어 가속기 시뮬레이터 (INT8)")
        
        # 1. 하드웨어 모델 초기화
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model = LeNet5HardwareSim(weight_dir=os.path.join(BASE_DIR, "weights_txt"))
        
        # 2. 그림판(Canvas) 설정 (280x280 크기, 검은 바탕)
        self.canvas = tk.Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack(pady=10)
        
        # 보이지 않는 메모리 상의 이미지 (추론용)
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        # 마우스를 드래그할 때 그림이 그려지도록 이벤트 연결
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # 3. 버튼 및 텍스트 설정
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        
        tk.Button(btn_frame, text="추론하기", command=self.predict, font=('Arial', 14)).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="지우기", command=self.clear, font=('Arial', 14)).pack(side=tk.LEFT, padx=10)
        
        self.label = tk.Label(root, text="Waiting for an input", font=('Arial', 16))
        self.label.pack(pady=15)
        
    def paint(self, event):
        # 펜 굵기 (MNIST 이미지와 비슷한 비율을 위해 12로 설정)
        r = 12 
        x1, y1 = event.x - r, event.y - r
        x2, y2 = event.x + r, event.y + r
        
        # 화면에 하얀색 동그라미를 연속으로 그려서 선을 만듦
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        # 메모리 상의 이미지에도 똑같이 그림
        self.draw.ellipse([x1, y1, x2, y2], fill="white")
        
    def clear(self):
        # 그림판 초기화
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Waiting for an input")
        
    def predict(self):
        # 1. 그려진 이미지를 LeNet-5 입력 크기(32x32)로 축소
        img_resized = self.image.resize((32, 32), Image.Resampling.LANCZOS)
        
        # 2. 0~255 픽셀 값을 0.0 ~ 1.0으로 변환
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # 3. 데이터 정규화 (MNIST 학습할 때 썼던 0.1307, 0.3081 사용)
        img_norm = (img_array - 0.1307) / 0.3081
        
        # 4. 하드웨어 입력을 위해 8비트 대칭 양자화(INT8) 변환
        max_val = np.max(np.abs(img_norm))
        s_img = max_val / 127.0 if max_val != 0 else 1e-8
        img_int8 = np.round(img_norm / s_img).astype(np.int32).reshape(1, 32, 32)
        
        # 5. 하드웨어 모델 추론 실행 (터미널에 데이터 흐름 자동 출력!)
        print("\n" + "="*45)
        print("[ 사용자 입력 이미지 실시간 하드웨어 추론 ]")
        
        out = self.model.forward(img_int8, s_img)
        pred = np.argmax(out)
        
        print(f"-> 칩 최종 예측 숫자: {pred}")
        print("="*45)
        
        # UI 업데이트
        self.label.config(text=f"Answer: {pred}", fg="blue")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()