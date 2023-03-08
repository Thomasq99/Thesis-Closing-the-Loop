import os
from Concepts.helper import load_images_from_files
import matplotlib.pyplot as plt
import random
import json
import numpy as np

with open('./data/ImageNet/Labels.json') as file:
    dct = json.load(file)

chosen_classes = random.sample(list(dct.keys()), k=10)

images = np.zeros((10, 5, 299, 299, 3))

for idx, class_ in enumerate(chosen_classes):
    train_folders = ['train.X1', 'train.X2', 'train.X3', 'train.X4']
    for folder in train_folders:
        if class_ in os.listdir(os.path.join('data/ImageNet', folder)):

            imgs_class_ = load_images_from_files([os.path.join('data/ImageNet', folder, class_, file) for file in
                                                  os.listdir(os.path.join('data/ImageNet', folder, class_))],
                                                 max_imgs=5, do_shuffle=True)
            images[idx] = imgs_class_
            break

fig, axs = plt.subplots(10, 5)

for i in range(10):
    imgs = images[i]
    class_ = dct[chosen_classes[i]].split(',')[0]

    fig, axs = plt.subplots(1, 5)
    axs = axs.flatten()
    for idx, img, ax in zip(range(5), imgs, axs):
        ax.imshow(img)
        ax.axis('off')
        if idx == 2:
            ax.set_title(dct[chosen_classes[i]])

    plt.savefig(f'Questionnaire_images/imgs_{class_}.png', dpi=700)


