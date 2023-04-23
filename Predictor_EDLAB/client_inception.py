import sys
import time

sys.path.append('../')
import torch
import torch.nn as nn

import Models as models
import numpy as np


model_name = str(np.load('m_n.npy'))
model_cfg = list(np.load('m_c.npy'))

time.sleep(3)
model = models.__dict__[model_name](model_cfg, init_weight=False)

model.cuda()
model.eval()

with torch.no_grad():
    image = torch.rand(1, 3, 299, 299)  # N* C*H*W
    image = image.cuda()
    model(image)
    torch.cuda.synchronize(torch.cuda.current_device())

    start = time.time()
    for i in range(20):
        model(image)
    torch.cuda.synchronize(torch.cuda.current_device())
    end = time.time()

lat = (end - start) / 20
print(lat)
del model
time.sleep(5)
