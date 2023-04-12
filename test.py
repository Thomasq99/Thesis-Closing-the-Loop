# import pickle as p
# from Concepts.ConceptBank import ConceptBank
# import matplotlib.pyplot as plt
# import os
#
# from Concepts.helper import load_images_from_files
#
# with open('saved_concept_bank.pkl', 'rb') as file:
#     concept_bank = p.load(file)
#
# CB = ConceptBank(concept_bank['concept_bank_dct']['mixed8'])
#
# fig = CB.plot_concept('userDefined_africanEar2.jpg', num_images=5, shape=(90, 90))
# fig.show()
# #
# # fig = CB.plot_concept('userDefined_africanEar1.jpg', num_images=5, shape=(90, 90))
# # fig.show()
# #
# # fig = CB.plot_concept('userDefined_trunk1.jpg', num_images=5, shape=(90, 90))
# # fig.show()
# #
# # fig = CB.plot_concept('userDefined_tusk1.jpg', num_images=5, shape=(90, 90))
# # fig.show()
# #
# # fig = CB.plot_concept('African_elephant__concept13', num_images=5, shape=(90, 90))
# # fig.show()
# #
# # fig = CB.plot_concept('African_elephant__concept5', num_images=5, shape=(90, 90))
# # fig.show()
#
# # class_folder = './questionnaire/indian_ear'
# # img_filenames = [os.path.join(class_folder, filename) for filename in os.listdir(class_folder)]
# # imgs = load_images_from_files(filenames=img_filenames, max_imgs=20, num_workers=1)[:5]
# # fig, axs = plt.subplots(1, 5)
# # axs = axs.flatten()
# # for idx, img, ax in zip(range(5), imgs, axs):
# #     ax.imshow(img)
# #     ax.axis('off')
# # fig.tight_layout()
# # fig.savefig(fname=f'subset_images/images_indian_ear.png', dpi=400)

from typing import List, Dict
from Concepts.CAV import CAV, get_or_train_cav
from Concepts.helper import load_images_from_files, get_gradients_of_images, get_grad_model, get_bottleneck_model, \
    get_activations_of_images
from Dash_helper import get_class_labels, read_image, get_random_activation, save_images
import os
import tensorflow as tf
import shutil
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # my GPU is too small to save enough images in its V-RAM to get the gradients

bottleneck = 'mixed8'

cav1 = CAV.load_cav('./ACE_output/ImageNet/cavs/mixed8-toucan__userDefined_001.pkl')
cav2 = CAV.load_cav('./ACE_output/ImageNet/cavs/mixed8-toucan__concept1.pkl')

class_folder = '../../data/ImageNet/n01843383'

class_dct = get_class_labels(data_folder_path='../../data/ImageNet')
img_filenames = [os.path.join(class_folder, filename) for filename in os.listdir(class_folder)]
imgs = load_images_from_files(filenames=img_filenames, num_workers=1)

cavs = [CAV.load_cav(f'./ACE_output/ImageNet/cavs_temp/mixed8-toucan__concept6-random500_{i}.pkl') for i in range(20)]

model = tf.keras.applications.inception_v3.InceptionV3()
bn_model = get_grad_model(model, bottleneck)
grads = get_gradients_of_images(imgs, bn_model, class_dct['toucan'])
tcav_scores = [cav.compute_tcav_score(grads) for cav in cavs]
print(tcav_scores)

# USER DEFINED
filenames = [os.path.join('ACE_output/ImageNet/concepts/toucan/mixed8/toucan__concept6', file) for file in os.listdir('ACE_output/ImageNet/concepts/toucan/mixed8/toucan__concept6')]
images = load_images_from_files(filenames)
def create_new_concept(images, session_dir: str, bottleneck: str, model: tf.keras.models.Model):

    concept_name = 'toucan_test'
    concept_dir = os.path.join(session_dir, 'concepts', concept_name)

    # create directory for storing the images of the user defined concept
    if not os.path.exists(concept_dir):
        os.makedirs(concept_dir)
    else:
        shutil.rmtree(concept_dir)
    os.makedirs(os.path.join(concept_dir, 'images'))

    # get activations of the random counterpart
    random_activations = get_random_activation(session_dir, bottleneck)

    # get activations of the images
    concept_activations = get_activations_of_images(images, get_bottleneck_model(model, bottleneck))

    # for each random counterpart create a CAV
    cavs = []
    for rnd_act in random_activations:
        act_dct = {concept_name: concept_activations.reshape((concept_activations.shape[0], -1)),
                   'random_counterpart': rnd_act.reshape((rnd_act.shape[0], -1))}
        cavs.append(get_or_train_cav([concept_name, 'random_counterpart'], bottleneck,
                                     os.path.join(session_dir, 'cavs'), act_dct, save=False, ow=True))
    tcav_scores = [cav.compute_tcav_score(grads) for cav in cavs]


    return tcav_scores

scores = create_new_concept(images, './ACE_output/ImageNet', 'mixed8', model)
print(scores)

#TODO error in loading images