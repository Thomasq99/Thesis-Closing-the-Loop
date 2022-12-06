from Concepts.ACE import ACE
import tensorflow as tf
import os
import shutil
from Concepts.ACE_helper import ace_create_source_dir_imagenet
# TODO in the end remove ace_create_source_dir_imagenet
from plotly.subplots import make_subplots
from skimage.segmentation import mark_boundaries
import numpy as np
import plotly.graph_objects as go

def prepare_ACE(model_name, source_data, working_dir, target_class, bottleneck_string, overwrite=False):
    # TODO write asserts
    # get bottlenecks
    bottlenecks = [string.strip() for string in bottleneck_string.split(',')]

    # load model
    if model_name == 'InceptionV3':
        model = tf.keras.applications.inception_v3.InceptionV3()
    elif os.path.exists(model_name):
        model = tf.keras.models.load_model(model_name)
    else:
        raise ValueError(f'{model_name} is not a directory to a model nor the InceptionV3model')

    # Create working directory
    if overwrite and os.path.exists(working_dir):
        shutil.rmtree(working_dir)

    discovered_concepts_dir = os.path.join(working_dir, 'concepts/')
    results_dir = os.path.join(working_dir, 'results/')
    cavs_dir = os.path.join(working_dir, 'cavs/')
    activations_dir = os.path.join(working_dir, 'acts/')
    results_summaries_dir = os.path.join(working_dir, 'results_summaries/')

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    if not os.path.exists(discovered_concepts_dir):
        os.makedirs(discovered_concepts_dir)
    else:
        shutil.rmtree(discovered_concepts_dir)
        os.makedirs(discovered_concepts_dir)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(cavs_dir):
        os.makedirs(cavs_dir)
    else:
        shutil.rmtree(cavs_dir)
        os.makedirs(cavs_dir)
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)
    if not os.path.exists(results_summaries_dir):
        os.makedirs(results_summaries_dir)


    # prepare data to the right format. Change this function for your own use.
    class_to_id = ace_create_source_dir_imagenet('./data/ImageNet', source_data, target_class, num_random_concepts=20,
                                                 ow=overwrite)

    print('data prepared')

    # initialize ACE
    ace = ACE(model, bottlenecks, target_class, source_data, activations_dir, cavs_dir, 'random_discovery',
              class_to_id, num_random_concepts=20, num_workers=30)

    return discovered_concepts_dir, ace


def plot_concepts(bottleneck, ace_obj):
    fig = make_subplots(2 * len(ace_obj.concept_dict[bottleneck]['concepts']), 10)

    for j, concept in enumerate(ace_obj.concept_dict[bottleneck]['concepts']):
        # TODO add support for different modes, add bottleneck chooser, optional add functionality for number of images
        concept_images = ace_obj.concept_dict[bottleneck][concept]['images']
        concept_patches = ace_obj.concept_dict[bottleneck][concept]['patches']
        concept_image_numbers = ace_obj.concept_dict[bottleneck][concept]['image_numbers']
        idxs = np.arange(len(concept_images))[:10]
        for i, idx in enumerate(idxs):
            fig.add_trace(go.Image(z=concept_images[idx] * 255), j+1, i + 1)
            mask = 1 - (np.mean(concept_patches[idxs[i]] == float(ace_obj.average_image_value) / 255, -1) == 1)
            image = ace_obj.discovery_images[concept_image_numbers[idx]]
            fig.add_trace(go.Image(z=mark_boundaries(image, mask, color=(1, 1, 0), mode='thick') * 255), (j+1)*2, i + 1)

    return fig
