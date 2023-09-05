import json
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from Concepts.helper import load_images_from_files
with open('data/ImageNet/imagenet_class_index.json') as file:
    dct = json.load(file)
#
# random.seed(76583)
# chosen_classes = random.sample(list(dct.keys()), k=10)
# images = np.zeros((10, 5, 299, 299, 3))
#
# for i in range(10):
#     class_folder = os.path.join('data/ImageNet', dct[chosen_classes[i]][0])
#     img_filenames = [os.path.join(class_folder, filename) for filename in os.listdir(class_folder)]
#     imgs = load_images_from_files(filenames=img_filenames, max_imgs=10, num_workers=1)
#     images[i] = imgs[:5]
#
# with open('questionnaire/questionnaire_images.npy', 'wb') as f:
#     np.save(f, images)
#
# # PLOT RANDOM CHOSEN CLASSES
# for i in range(10):
#     fig, axs = plt.subplots(1, 5)
#     axs = axs.flatten()
#     for idx, img, ax in zip(range(5), images[i], axs):
#         ax.imshow(img)
#         ax.axis('off')
#         if idx == 2:
#             ax.set_title(dct[chosen_classes[i]][1])
#     fig.tight_layout()
#     fig.savefig(fname=f'questionnaire/questionnaire_images_{dct[chosen_classes[i]][1]}.png', dpi=400)
#


# EXPERIMENT 2 WITH SUBSETS

def plot_imgs_class(target_class):
    with open('data/ImageNet/imagenet_class_index.json') as file:
        dct = json.load(file)

    for key, value in dct.items():
        if value[1] == target_class:
            class_folder = os.path.join('data/ImageNet', value[0])

    img_filenames = [os.path.join(class_folder, filename) for filename in os.listdir(class_folder)]
    imgs = load_images_from_files(filenames=img_filenames, max_imgs=20, num_workers=1)
    imgs = imgs[:10]
    fig, axs = plt.subplots(2, 5)
    axs = axs.flatten()
    for idx, img, ax in zip(range(10), imgs, axs):
        ax.imshow(img)
        ax.axis('off')
        if idx == 2:
            ax.set_title(target_class)
    fig.tight_layout()
    fig.savefig(fname=f'subset_images/images_{target_class}.png', dpi=400)
if not os.path.exists('./subset_images'):
    os.makedirs('./subset_images')
plot_imgs_class('stingray')
# plot_imgs_class('electric_ray')
# plot_imgs_class('toucan')
# plot_imgs_class('hornbill')
# plot_imgs_class('snail')
# plot_imgs_class('slug')
# plot_imgs_class('hermit_crab')
# plot_imgs_class('rock_crab')
# plot_imgs_class('Dungeness_crab')
# plot_imgs_class('American_lobster')
# plot_imgs_class('spiny_lobster')
# plot_imgs_class('red_fox')
# plot_imgs_class('kit_fox')
# plot_imgs_class('Arctic_fox')
# plot_imgs_class('grey_fox')
# plot_imgs_class('wild_boar')
# plot_imgs_class('warthog')
# plot_imgs_class('Indian_elephant')
# plot_imgs_class('African_elephant')

