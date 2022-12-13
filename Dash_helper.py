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
from PIL import Image


def prepare_output_directories(working_dir, target_class, bottlenecks, overwrite=False):
    # create directories
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
        if not os.path.exists(os.path.join(discovered_concepts_dir, target_class)):
            os.makedirs(os.path.join(discovered_concepts_dir, target_class))

    for bottleneck in bottlenecks:
        if not os.path.exists(os.path.join(discovered_concepts_dir, target_class, bottleneck)):
            os.makedirs(os.path.join(discovered_concepts_dir, target_class, bottleneck))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(cavs_dir):
        os.makedirs(cavs_dir)

    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)

    if not os.path.exists(results_summaries_dir):
        os.makedirs(results_summaries_dir)

    return activations_dir, cavs_dir, discovered_concepts_dir


def prepare_ACE(model_name, source_data, working_dir, target_class, bottlenecks, overwrite=False):
    # TODO write asserts

    # prepare output directories
    activations_dir, cavs_dir, discovered_concepts_dir = prepare_output_directories(working_dir, target_class,
                                                                                    bottlenecks, overwrite)

    # load model
    if model_name == 'InceptionV3':
        model = tf.keras.applications.inception_v3.InceptionV3()
    elif os.path.exists(model_name):
        model = tf.keras.models.load_model(model_name)
    else:
        raise ValueError(f'{model_name} is not a directory to a model nor the InceptionV3model')


    # prepare data to the right format. Change this function for your own use.
    class_to_id = ace_create_source_dir_imagenet('./data/ImageNet', source_data, target_class, num_random_concepts=20,
                                                 ow=False)

    print('data prepared')

    # initialize ACE
    ace = ACE(model, bottlenecks, target_class, source_data, working_dir, 'random_discovery',
              class_to_id, num_random_concepts=20, num_workers=30)

    return discovered_concepts_dir, ace
