import random

from Concepts.ACE import ACE
import os
import shutil
from Concepts.helper import save_concepts, save_images, load_images_from_files, \
    get_activations_of_images, get_bottleneck_model
from Concepts.ConceptBank import ConceptBank
# TODO in the end remove ace_create_source_dir_imagenet
import numpy as np
import base64
from pathlib import Path
import skimage.io
import tempfile
from Concepts.CAV import get_or_train_cav
from PIL import Image
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import tensorflow as tf
import json

def prepare_output_directories(working_dir: str, target_class: str, bottlenecks: List,
                               overwrite: bool = False) -> Tuple[str, str, str]:
    """ Prepares (makes/clears) all output directories for the current session.
    If overwrite is True, existing directories will be cleared.

    @param working_dir: Name of the folder of the current session. All future folders will be in this session folder.
    @param target_class: Name of the target class for which concepts will be found.
    @param bottlenecks: List of the names of the bottleneck layers for which concepts will be computed.
    @param overwrite: If True, overwrite existing directories.
    @return: folder names of the following: activations, cavs, concepts.
    """
    # create session directory
    if overwrite and os.path.exists(working_dir):
        shutil.rmtree(working_dir)

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    # create remaining directories
    discovered_concepts_dir = os.path.join(working_dir, 'concepts/')
    results_dir = os.path.join(working_dir, 'results/')
    cavs_dir = os.path.join(working_dir, 'cavs/')
    activations_dir = os.path.join(working_dir, 'acts/')
    results_summaries_dir = os.path.join(working_dir, 'results_summaries/')

    if not os.path.exists(discovered_concepts_dir):
        os.makedirs(discovered_concepts_dir)
    else:
        if not os.path.exists(os.path.join(discovered_concepts_dir, target_class)):
            os.makedirs(os.path.join(discovered_concepts_dir, target_class))

    for bottleneck in bottlenecks:
        if not os.path.exists(os.path.join(discovered_concepts_dir, target_class, bottleneck)):
            os.makedirs(os.path.join(discovered_concepts_dir, target_class, bottleneck))

    for directory in [results_dir, cavs_dir, activations_dir, results_summaries_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    return activations_dir, cavs_dir, discovered_concepts_dir


def prepare_data(data_folder, num_random_concepts=20):
    # TODO REFACTOR THIS DCT
    with open('data/ImageNet/imagenet_class_index.json') as file:
        dct = json.load(file)

    class_to_id = {}
    id_to_folder = {}
    for key, value in dct.items():
        id_to_folder[int(key)] = value[0]
        class_to_id[value[1]] = int(key)

    classes = [value[0] for value in dct.values()]
    # make random concepts:
    for folder in (['random_discovery'] + [f'random500_{i}' for i in range(num_random_concepts)]):
        folder_name = os.path.join(data_folder, folder)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            chosen_class_folders = random.choices(classes, k=500)
            filenames = [os.path.join(data_folder, class_folder,
                                      random.choice(os.listdir(os.path.join(data_folder, class_folder))))
                         for class_folder in chosen_class_folders]
            for file in filenames:
                shutil.copyfile(file, os.path.join(folder_name, os.path.basename(file)))
    return class_to_id, id_to_folder


def prepare_ACE(model_name: str, source_data: str, working_dir: str, target_class: str, bottlenecks: List,
                overwrite: bool = False) -> Tuple[str, 'ACE']:
    """ Prepares directories and data necessary for running ACE. Also initializes an ACE object

    @param model_name: Name of the model.
    @param source_data: Name of the directory containing the source images.
    @param working_dir: Name of the session directory.
    @param target_class: Name of the class for which concepts need to be found.
    @param bottlenecks: List of the bottleneck names for which concepts will be computed.
    @param overwrite: If True, overwrite existing directories.
    @return: Tuple consisting of (Name of the directory where concepts are stored, ACE object)
    """

    # Create and, if wanted, clear necessary directories
    activations_dir, cavs_dir, discovered_concepts_dir = prepare_output_directories(working_dir, target_class,
                                                                                    bottlenecks, overwrite)

    # IMPORTANT prepare data to the right format. Change this function for your own use.
    class_to_id, id_to_folder = prepare_data(data_folder=source_data, num_random_concepts=20)

    # initialize ACE
    ace = ACE(model_name, bottlenecks, target_class, source_data, working_dir, 'random_discovery',
              class_to_id, id_to_folder= id_to_folder, num_random_concepts=20, num_workers=30)

    return discovered_concepts_dir, ace


def run_ACE(model_name: str, path_to_source: str, path_to_working_dir: str, target_class: str, bottlenecks: List,
            concept_bank_dct: Dict, loaded_classes: List, mode: str = 'max') -> Tuple[Dict, List, bool]:
    """ Runs the ACE algorithm. Finding concepts for the target_class for each bottleneck in bottlenecks. If concepts
    are already computed previously, they will be loaded instead of computed.
    Concepts already loaded in current session will be skipped.

    @param model_name: Name of the tensorflow model or path to the tensorflow model.
    @param path_to_source: Name of the source directory where the images are stored.
    @param path_to_working_dir: Name of the session directory.
    @param target_class: Name of the class for which the concepts will be found.
    @param bottlenecks: List of bottlenecks for which the concepts will be computed.
    @param concept_bank_dct: Dictionary representing the current Concept Bank.
    @param loaded_classes: List of the bottleneck, classes combinations that are already loaded
    @param mode: Name of the aggregation method of the CAVs. Can be 'max' or 'average'.
        'max': Takes the CAV computed against the random counterpart which gives the highest accuracy
        'average': Averages the CAV direction of all CAVs trained against each random counterpart.
    @return: Updated concept_bank, updated list of computed bottleneck, target_class combinations, True
    """

    # check which bottleneck, class combinations are loaded
    loaded_classes_temp = [stored.split(', ') for stored in loaded_classes]
    loaded_bottlenecks = [bottleneck for bottleneck, class_ in loaded_classes_temp if class_ == target_class]
    bottlenecks = list(set(bottlenecks) - set(loaded_bottlenecks))

    if not bottlenecks:  # If all bottlenecks are already stored in this session, no update is needed
        print(f'all bottlenecks are already loaded in for {target_class}')
        return concept_bank_dct, loaded_classes, False

    # for each bottleneck that needs computing run ACE
    for bn in bottlenecks:
        loaded_classes.append(f'{bn}, {target_class}')

    discovered_concepts_dir, ace = prepare_ACE(model_name, path_to_source, path_to_working_dir,
                                               target_class, bottlenecks)
    image_dir = os.path.join(discovered_concepts_dir, target_class, 'images')

    # check which bottlenecks are not yet stored somewhere
    bn_to_find_concepts_for = [bn for bn in bottlenecks if not os.listdir(
        os.path.join(discovered_concepts_dir, target_class, bn))]
    bn_precomputed_concepts = list(set(bottlenecks) - set(bn_to_find_concepts_for))

    # LOAD EXISTING CONCEPTS
    if bn_precomputed_concepts:
        for bn in bn_precomputed_concepts:
            print(f'loading in concepts for {bn}')
            concepts = os.listdir(os.path.join(discovered_concepts_dir, target_class, bn))
            concepts = [concept for concept in concepts if not concept.endswith('patches')]
            if bn in concept_bank_dct:
                cb = ConceptBank(concept_bank_dct[bn])
                cb.add_concept(concepts)
                concept_bank_dct[bn] = cb
            else:
                concept_bank_dct[bn] = ConceptBank(dict(bottleneck=bn, working_dir=path_to_working_dir,
                                                        concept_names=concepts, class_id_dct=ace.class_to_id,
                                                        model_name=model_name))

    # FIND NEW CONCEPTS
    if bn_to_find_concepts_for:  # if not empty discover concepts for bottlenecks
        print(f'discovering concepts for {bn_to_find_concepts_for}')
        print('Creating patches')
        ace.bottlenecks = bn_to_find_concepts_for

        # STEP 1: SEGMENTATION OF IMAGES
        if os.path.exists(image_dir):  # if exists, patches are already created
            ace.create_patches_for_data(discovery_images=load_images_from_files(
                [os.path.join(image_dir, file) for file in os.listdir(image_dir)]))
        else:
            # create patches
            os.makedirs(image_dir)
            ace.create_patches_for_data()
            save_images(image_dir,
                        (ace.discovery_images * 255).astype(np.uint8))  # save images used for creating patches

        print('Discover concepts')

        # STEP 2: CLUSTERING SEGMENTS
        ace.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
        del ace.dataset  # Free memory
        del ace.image_numbers
        del ace.patches

        # Save discovered concept images (resized and original sized)
        save_concepts(ace, os.path.join(discovered_concepts_dir, target_class))

        print('Calculating CAVs')
        # STEP 3: CREATING CAVS
        accuracies = ace.cavs()

        print('combining CAVs')
        ace.aggregate_cavs(accuracies, mode=mode)

        concept_dict = {bn: ace.concept_dict[bn]['concepts'] for bn in ace.concept_dict.keys()}

        for bn in concept_dict.keys():
            if bn in concept_bank_dct:
                cb = ConceptBank(concept_bank_dct[bn])
                cb.add_concept(concept_dict[bn])
                concept_bank_dct[bn] = cb
            else:
                concept_bank_dct[bn] = ConceptBank(dict(bottleneck=bn, working_dir=path_to_working_dir,
                                                        concept_names=concept_dict[bn], class_id_dct=ace.class_to_id,
                                                        model_name=ace.model_name))

    del ace
    return concept_bank_dct, loaded_classes, True


def has_valid_image_extension(filename: str) -> bool:
    """ Checks whether the current filename has a valid image extension. One of [.png, .jpeg, .jpg] are accepted

    @param filename: Name of the image file
    @return True if the image extension is valid, False otherwise.
    """
    return Path(filename).suffix in (".png", ".jpeg", ".jpg")


def read_image(upload_contents: str, filename: str, shape: Tuple = (299, 299)) -> np.ndarray:
    """
    Read a base64 uploaded image using `skimage.io.imread` into a numpy array. Resizes it to be the same shape as
    the shape parameter

    @param upload_contents: base64 encoded string of the image.
    @param filename: Name of the image file.
    @param shape: Tuple denoting the required shape of the image.
    @return img: Image in array format.
    """
    # decode the image
    content_type, content_string = upload_contents.split(",")
    decoded = base64.b64decode(content_string)
    extension = Path(filename).suffix

    if not has_valid_image_extension(filename):
        raise ValueError('Uploaded file cannot be opened as an image')

    # `imread` expects a file, so we create one in a temporary directory
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp).joinpath("file").with_suffix(extension)
        file.write_bytes(decoded)
        img = skimage.io.imread(file)
        img = np.array(Image.fromarray(img).convert('RGB').resize(shape))
        return img


def get_random_activation(session_dir: str, bottleneck: str) -> List:
    """ Loads and returns the activations of the random counterparts.

    @param session_dir: Name of the current session of the user
    @param bottleneck: Name of the bottleneck for which to find the activations
    @return: List with the activations of the random counterparts
    """
    random_activations = os.listdir(os.path.join(session_dir, 'acts'))

    return [np.load(os.path.join(session_dir, 'acts', rnd_acts_path)).squeeze() for rnd_acts_path in random_activations
            if bottleneck in rnd_acts_path]


def create_new_concept(list_of_contents: List, filenames: List, session_dir: str, bottleneck: str,
                       model: tf.keras.models.Model, mode: str = 'max'):
    """ Creates a new CAV representing a concept from a list of images. Creates a new CAV trained on each random
        counterpart. The user can either take the maximum CAV accuracy or take the average of all CAVs. The found CAV
        will be stored, so it can be loaded later.

    @param list_of_contents: List of base64 format images.
    @param filenames: List of filenames of the images.
    @param session_dir: Directory of the session the current user is in.
    @param bottleneck: Name of the bottleneck layer of the model for which the concept will be created.
    @param model: Tensorflow model for which the concept will be found.
    @param mode: Mode of averaging the concepts. Can be 'max' or 'average'.
        'max': will take the concept with maximum CAV accuracy.
        'average': will average the learned CAVs out.
    @return: concept_name: The name of the learned concept.
    """
    # read in images
    images = np.stack([read_image(content, filename) for content, filename in zip(list_of_contents, filenames)], axis=0)
    concept_name = f'userDefined_{filenames[0].split("_")[0]}'

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

    if mode == 'max':  # save CAV with maximum accuracy
        cav_accuracies = [cav.accuracy for cav in cavs]
        max_cav = cavs[np.argmax(cav_accuracies)]
        max_cav.file_name = f'{max_cav.bottleneck}-{max_cav.concept}.pkl'
        max_cav.save_cav(os.path.join(session_dir, 'cavs'))

    elif mode == 'average':  # save average of all CAV directions
        cav_avg = cavs[0]
        cav_avg.cav = np.mean(np.array([cav.cav for cav in cavs]), axis=0)
        cav_avg.file_name = f'{cav_avg.bottleneck}-{cav_avg.concept}.pkl'
        cav_avg.save_cav(os.path.join(session_dir, 'cavs'))

    # save images used to train the CAV
    save_images(os.path.join(concept_dir, 'images'), list(images))

    return concept_name


def get_sessions_concepts(concept_bank_dct: Dict) -> List:
    """ Returns the concept names of all concepts found or defined in the user's current session

    @param concept_bank_dct: The Concept Bank in dictionary format. {bottleneck: {concept_names:[names], ...}, ...}
    @return concepts: List of the concepts present in current session. In the following format
        [{value: concept_name1, label:concept_name1}, ...]
    """
    concepts = []
    for bn in concept_bank_dct:
        for concept in concept_bank_dct[bn]['concept_names']:
            label = f'{bn}, {concept}'
            concepts.append({'value': label, 'label': label})
    return concepts


def get_sessions_bottlenecks(concept_bank_dct: Dict) -> List:
    """ Returns the bottlenecks for which concepts are found in the user's current session

    @param concept_bank_dct: The Concept Bank in dictionary format. {bottleneck: ConceptBank_dct, ...}
    """
    # update options for bottleneck for visualizations
    bottlenecks = list(concept_bank_dct.keys())
    bottleneck_options = [{'value': bn, 'label': bn} for bn in bottlenecks]
    return bottleneck_options


def blank_fig() -> go.Figure:
    """ function that returns a blank figure

    @return blank figure."""
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    return fig


def get_class_labels(data_folder_path: str):
    """ Gets the class_to_id dictionary from the data folder

    @param data_folder_path: path to the data folder
    @return: dictionary mapping classes to the id's
    """
    with open(os.path.join(data_folder_path, 'imagenet_class_index.json')) as jsonFile:
        imagenet_dct = json.load(jsonFile)
        jsonFile.close()

    class_to_id = {}
    id_to_folder = {}
    for key, value in imagenet_dct.items():
        id_to_folder[int(key)] = value[0]
        class_to_id[value[1]] = int(key)
    return class_to_id


def load_model(model_name):
    if model_name == 'InceptionV3':
        model = tf.keras.applications.inception_v3.InceptionV3()
    elif os.path.exists(model_name):
        model = tf.keras.models.load_model(model_name, compile=False)
    else:
        raise ValueError(f'{model_name} is not a directory to a model nor the InceptionV3model')
    return model
