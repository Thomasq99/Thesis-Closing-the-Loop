from Concepts.ACE import ACE
import os
import shutil
from Concepts.helper import ace_create_source_dir_imagenet, save_concepts, save_images, load_images_from_files, \
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


    # prepare data to the right format. Change this function for your own use.
    class_to_id = ace_create_source_dir_imagenet('./data/ImageNet', source_data, target_class, num_random_concepts=20,
                                                 ow=False)

    print('data prepared')

    # initialize ACE
    ace = ACE(model_name, bottlenecks, target_class, source_data, working_dir, 'random_discovery',
              class_to_id, num_random_concepts=20, num_workers=30)

    return discovered_concepts_dir, ace


def run_ACE(model_name, path_to_source, path_to_working_dir, target_class, bottlenecks, concept_bank_dct,
            loaded_classes, mode='max'):

    # check which bottlenecks are loaded
    loaded_classes_temp = [stored.split(', ') for stored in loaded_classes]
    loaded_bottlenecks = [bottleneck for bottleneck, class_ in loaded_classes_temp if class_ == target_class]
    bottlenecks = list(set(bottlenecks) - set(loaded_bottlenecks))
    if not bottlenecks:  # see if empty
        print(f'all bottlenecks are already loaded in for {target_class}')
        return concept_bank_dct, loaded_classes, False

    for bn in bottlenecks:
        loaded_classes.append(f'{bn}, {target_class}')

    discovered_concepts_dir, ace = prepare_ACE(model_name, path_to_source, path_to_working_dir,
                                               target_class, bottlenecks)

    # find if patches are already created once
    image_dir = os.path.join(discovered_concepts_dir, target_class, 'images')

    # check which bottlenecks are not yet created
    bn_to_find_concepts_for = [bn for bn in bottlenecks if not os.listdir(
        os.path.join(discovered_concepts_dir, target_class, bn))]
    bn_precomputed_concepts = list(set(bottlenecks) - set(bn_to_find_concepts_for))

    #######################################################################
    ################### Load existing concepts ############################
    #######################################################################
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

    #######################################################################
    ################### Find new concepts #################################
    #######################################################################

    if bn_to_find_concepts_for:  # if not empty discover concepts for bottlenecks
        print(f'discovering concepts for {bn_to_find_concepts_for}')
        print('Creating patches')
        ace.bottlenecks = bn_to_find_concepts_for
        if os.path.exists(image_dir):
            ace.create_patches_for_data(discovery_images=load_images_from_files(
                [os.path.join(image_dir, file) for file in os.listdir(image_dir)]))
        else:
            os.makedirs(image_dir)
            ace.create_patches_for_data()
            save_images(image_dir,
                        (ace.discovery_images * 255).astype(np.uint8))  # save images used for creating patches

        print('Discover concepts')
        ace.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
        del ace.dataset  # Free memory
        del ace.image_numbers
        del ace.patches

        # Save discovered concept images (resized and original sized)
        save_concepts(ace, os.path.join(discovered_concepts_dir, target_class))

        print('Calculating CAVs')
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


def has_valid_image_extension(filename):
    return Path(filename).suffix in (".png", ".jpeg", ".jpg")


def read_image(upload_contents, filename, shape=(299, 299)):
    """
    Read a base64 uploaded image using `skimage.io.imread` into a numpy array.
    """
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


def get_random_activation(session_dir, bottleneck):
    random_activations = os.listdir(os.path.join(session_dir, 'acts'))

    return [np.load(os.path.join(session_dir, 'acts', rnd_acts_path)).squeeze() for rnd_acts_path in random_activations
            if bottleneck in rnd_acts_path]


def create_new_concept(list_of_contents, filenames, session_dir, bottleneck, model, mode='max'):
    images = np.stack([read_image(content, filename) for content, filename in zip(list_of_contents, filenames)], axis=0)
    concept_name = f'userDefined_{filenames[0].split("_")[0]}'

    concept_dir = os.path.join(session_dir, 'concepts', concept_name)

    if not os.path.exists(concept_dir):
        os.makedirs(concept_dir)
    else:
        shutil.rmtree(concept_dir)

    os.makedirs(os.path.join(concept_dir, 'images'))

    random_activations = get_random_activation(session_dir, bottleneck)
    concept_activations = get_activations_of_images(images, get_bottleneck_model(model, bottleneck))
    cavs = []
    for rnd_act in random_activations:
        act_dct = {concept_name: concept_activations.reshape((concept_activations.shape[0], -1)),
                   'random_counterpart': rnd_act.reshape((rnd_act.shape[0], -1))}
        cavs.append(get_or_train_cav([concept_name, 'random_counterpart'], bottleneck,
                                     os.path.join(session_dir, 'cavs'), act_dct, save=False, ow=True))

    if mode == 'max':
        cav_accuracies = [cav.accuracy for cav in cavs]
        max_cav = cavs[np.argmax(cav_accuracies)]
        max_cav.file_name = f'{max_cav.bottleneck}-{max_cav.concept}.pkl'
        max_cav.save_cav(os.path.join(session_dir, 'cavs'))

    elif mode == 'average':
        cav_avg = cavs[0]
        cav_avg.cav = np.mean(np.array([cav.cav for cav in cavs]), axis=0)
        cav_avg.file_name = f'{cav_avg.bottleneck}-{cav_avg.concept}.pkl'
        cav_avg.save_cav(os.path.join(session_dir, 'cavs'))

    save_images(os.path.join(concept_dir, 'images'), list(images))

    return concept_name


def update_remove_options(concept_bank_dct):
    # update options for removing concept
    remove_options = []
    for bn in concept_bank_dct:
        for concept in concept_bank_dct[bn]['concept_names']:
            label = f'{bn}, {concept}'
            remove_options.append({'value': label, 'label': label})
    return remove_options


def update_bottleneck_options(concept_bank_dct):
    # update options for bottleneck for visualizations
    bottlenecks = list(concept_bank_dct.keys())
    bottleneck_options = [{'value': bn, 'label': bn} for bn in bottlenecks]
    return bottleneck_options
