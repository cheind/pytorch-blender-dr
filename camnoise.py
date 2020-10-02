import albumentations as A
import matplotlib.pyplot as plt
import cv2
import os


f = "./test.png"
image = cv2.imread(f)
orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)

plt.axis('off')
plt.tight_layout()

# low = 10
# for step, upper in enumerate((20, 30, 40, 50, 60, 70, 80)):
#     trans = A.GaussNoise(p=1, var_limit=(low, upper))
#     low += 10
#     """var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
#     |          will be (0, var_limit). Default: (10.0, 50.0).
#     |      mean (float): mean of the noise. Default: 0
#     """
#     image = trans(image=orig_image)["image"]
#     plt.title(f"{low} - {upper}")
#     plt.imshow(image)
#     plt.savefig(f"testfig{step}.png")

low = 10
upper = 60
step = 10
trans = A.GaussNoise(p=1, var_limit=(low, upper))

for _ in range(10):
    image = trans(image=orig_image)["image"]
    plt.title(f"{low} - {upper}")
    plt.imshow(image)
    plt.savefig(f"testfig{step}.png")
    step += 1
