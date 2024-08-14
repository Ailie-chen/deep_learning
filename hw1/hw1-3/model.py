import numpy as np

def relu(x):
    return np.maximum(0, x)

class FCLayer:
    def __init__(self, input_dim, output_dim, init="he-uniform"):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 参数初始化
        std = (2. / input_dim) ** 0.5
        bound = (3 ** 0.5) * std
        if init == "he-uniform":
            self.W = (np.random.rand(input_dim, output_dim) - 0.5) * 2 * bound
        elif init == "he-gauss":
            self.W =  np.random.randn(input_dim, output_dim) * std
        elif init == "gauss":
            self.W = np.random.randn(input_dim, output_dim) * 0.02
        self.b = np.zeros(output_dim, dtype=np.float32)

        self.W_grad = np.zeros_like(self.W, dtype=np.float32)
        self.b_grad = np.zeros_like(self.b, dtype=np.float32)
    
    def forward(self, input):
        self.input = input
        self.z = np.dot(input, self.W) + self.b     # 未激活的输出
        self.a = np.maximum(self.z, 0)              # 激活值
        return self.a

    def backward(self, output_grad):
        output_grad[self.z<0] = 0       # 激活值对z求导       # FC(100, 10) 100 x 32 x 32 x 10 -> 100 x 10
        self.W_grad = np.dot(self.input.T, output_grad)     # input=32 x 100; output_grad=32 x 10
        self.b_grad = np.sum(output_grad, axis=0)
        return np.dot(output_grad, self.W.T)    # output_grad2 x W; output_grad1 x W

class NeuralNetwork:
    def __init__(self, input_size, output_size, layer_sizes, init="he-uniform"):
        self.layers = []
        prev_size = input_size
        for i in range(len(layer_sizes)):
            layer = FCLayer(prev_size, layer_sizes[i], init=init)
            prev_size = layer_sizes[i]
            self.layers.append(layer)
        self.layers.append(FCLayer(prev_size, output_size, init=init))   # 输出层

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
