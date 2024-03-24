import torch
import fcinasm
import numpy as np
import io

model = fcinasm.NeuralNetwork().to(fcinasm.device)
model.load_state_dict(torch.load(fcinasm.save_path))

bin: bytearray = bytearray()
for name, param in model.named_parameters():
    print(param)
    bin += param.data.flatten().numpy().tobytes()

f = io.FileIO(fcinasm.raw_parameters_save_path, "wb")
f.write(bin)