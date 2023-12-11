import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

def visualize(image, name):
    plt.figure(figsize=(20, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(name)
    
image = cv2.imread('images/rgb_00000.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

visualize(image, 'images/rgb.jpg')
    
transform = A.Compose(
    [A.RandomFog(fog_coef_lower=0.6, fog_coef_upper=0.7, alpha_coef=0.01, p=1)],
)
random.seed(7)
transformed = transform(image=image)
visualize(transformed['image'], 'images/transformed_image.jpg')
    