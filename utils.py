'''Some helper functions for PyTorch, including:
'''

import torch.nn as nn
import albumentations as A
from pytorch_grad_cam import GradCAM
import matplotlib.pyplot as plt


train_transform = A.Compose([
    A.Normalize(
        mean=(0.49139968, 0.48215841, 0.44653091),
           std=(0.24703223, 0.24348513, 0.26158784),
    ),
    A.RandomCrop(width=256, height=256),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_height=16, min_width=16, fill_value=(0.49139968, 0.48215841, 0.44653091), mask_fill_value=None)
])

test_transform = A.Compose([
    A.Normalize(
        mean=[0.49139968, 0.48215841, 0.44653091],
        std=[0.24703223, 0.24348513, 0.26158784],
    )
])



def show_images(images, r, c):
    figure = plt.figure(figsize=(14, 10))
    for i in range(1, c * r + 1):
        img = images[i]

        figure.add_subplot(r, c, i)
        plt.axis("off")
        plt.imshow(img, cmap="gray")

    plt.tight_layout()
    plt.show()