import torch 
from model import Model

device = torch.device('cpu')

B = 4
model = Model(10, 6).to(device)

x = torch.randn(B, 1, 12, 16, device=device)
state = torch.randn(B, 10, device=device)
h = None

act, values, h = model(x, state, h)

loss = act.pow(2).mean()
loss.backward()

print("act:", act.shape)
print("loss:", float(loss))
print("CPU forward/backward ok")
