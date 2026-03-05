import numpy as np
import os

class LeNet5NumPy:
    def __init__(self, weight_dir="weights_txt"):
        self.weights = {}
        # 텍스트 파일로부터 데이터 로드 및 리셰이프
        self._load_param(f"{weight_dir}/conv1.weight.txt", (6, 1, 5, 5), "c1_w")
        self._load_param(f"{weight_dir}/conv1.bias.txt", (6,), "c1_b")
        self._load_param(f"{weight_dir}/conv2.weight.txt", (16, 6, 5, 5), "c2_w")
        self._load_param(f"{weight_dir}/conv2.bias.txt", (16,), "c2_b")
        self._load_param(f"{weight_dir}/fc1.weight.txt", (120, 400), "f1_w")
        self._load_param(f"{weight_dir}/fc1.bias.txt", (120,), "f1_b")
        self._load_param(f"{weight_dir}/fc2.weight.txt", (84, 120), "f2_w")
        self._load_param(f"{weight_dir}/fc2.bias.txt", (84,), "f2_b")
        self._load_param(f"{weight_dir}/fc3.weight.txt", (10, 84), "f3_w")
        self._load_param(f"{weight_dir}/fc3.bias.txt", (10,), "f3_b")

    def _load_param(self, path, shape, name):
        data = np.loadtxt(path)
        self.weights[name] = data.reshape(shape)

    def tanh(self, x):
        return np.tanh(x)

    def avg_pool(self, x, kernel_size=2, stride=2):
        C, H, W = x.shape
        out_h, out_w = H // stride, W // stride
        out = np.zeros((C, out_h, out_w))
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    out[c, i, j] = np.mean(x[c, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size])
        return out

    def conv2d(self, x, weight, bias):
        out_c, in_c, k_h, k_w = weight.shape
        _, in_h, in_w = x.shape
        out_h, out_w = in_h - k_h + 1, in_w - k_w + 1
        out = np.zeros((out_c, out_h, out_w))
        
        for oc in range(out_c):
            for ic in range(in_c):
                for i in range(out_h):
                    for j in range(out_w):
                        # 하드웨어의 MAC(Multiply-Accumulate) 연산과 동일한 부분
                        out[oc, i, j] += np.sum(x[ic, i:i+k_h, j:j+k_w] * weight[oc, ic])
            out[oc] += bias[oc]
        return out

    def forward(self, x, debug=False):
        # x shape: (1, 32, 32)
        x = self.tanh(self.conv2d(x, self.weights['c1_w'], self.weights['c1_b']))
        if debug: print(f"C1 Output: {x.shape}")
        
        x = self.avg_pool(x)
        if debug: print(f"S2 Output: {x.shape}")
        
        x = self.tanh(self.conv2d(x, self.weights['c2_w'], self.weights['c2_b']))
        if debug: print(f"C3 Output: {x.shape}")
        
        x = self.avg_pool(x)
        if debug: print(f"S4 Output: {x.shape}")
        
        x = x.flatten()
        if debug: print(f"Flatten: {x.shape}")
        
        x = self.tanh(np.dot(self.weights['f1_w'], x) + self.weights['f1_b'])
        x = self.tanh(np.dot(self.weights['f2_w'], x) + self.weights['f2_b'])
        x = np.dot(self.weights['f3_w'], x) + self.weights['f3_b']
        if debug: print(f"Final Output: {x.shape}")
        
        return x