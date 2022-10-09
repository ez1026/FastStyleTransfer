import torch
from fast_style_transfer import TransNet, load_image
import numpy as np
import matplotlib.pyplot as plt
import cv2

g_net = TransNet()
model_data = torch.load("./fst.pth")
g_net.load_state_dict(model_data)

test_img = load_image("/mnt/d/temp/test.jpg")
shape = cv2.imread("/mnt/d/temp/test.jpg").shape[:2]
print(shape)
height, width = shape
demo = g_net(test_img)[0].detach()
demo = demo.numpy()
demo = np.transpose(demo, (1,2,0))
demo = cv2.resize(demo, dsize=(width, height))
print(demo.shape)
plt.imshow(demo)
plt.show()
