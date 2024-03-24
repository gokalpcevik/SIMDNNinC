import torch
from torchvision.transforms import ToTensor
import fcinasm
import time

start = time.time_ns()
model = fcinasm.NeuralNetwork().to(fcinasm.device)
model.load_state_dict(torch.load(fcinasm.save_path))

test_data = fcinasm.test_data

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
with torch.no_grad():
    for i in range(0, 1):
        x, y = test_data[i][0], test_data[i][1]
        x = x.to(fcinasm.device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        end = time.time_ns()
        print(f'Predicted: "{predicted}", Actual: "{actual}"\nTook {(end - start) / 10**3}us')
