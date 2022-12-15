from Concepts.ACE import ACE
import os
import shutil
from Concepts.helper import ace_create_source_dir_imagenet, save_concepts, save_images, load_images_from_files, \
    plot_concepts_matplotlib
from Concepts.ConceptBank import ConceptBank
# TODO in the end remove ace_create_source_dir_imagenet
import numpy as np


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


def run_ACE(model_name, path_to_source, path_to_working_dir, target_class, bottlenecks, mode='max'):

    discovered_concepts_dir, ace = prepare_ACE(model_name, path_to_source, path_to_working_dir,
                                               target_class, bottlenecks)

    # find if patches are already created once
    image_dir = os.path.join(discovered_concepts_dir, target_class, 'images')
    # TODO see if it can be empty
    concept_bank_dct = {}

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
            concept_bank_dct[bn] = ConceptBank(dict(bottleneck=bn, working_dir=path_to_working_dir,
                                                    concept_names=concepts, class_id_dct=ace.class_to_id,
                                                    model_name=ace.model_name))

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

        for bn in bn_to_find_concepts_for:
            plot_concepts_matplotlib(ace, bn, ace.target_class, address=os.path.join(path_to_working_dir, 'results/'))

        print('combining CAVs')
        ace.aggregate_cavs(accuracies, mode=mode)

        concept_dict = {bn: ace.concept_dict[bn]['concepts'] for bn in ace.concept_dict.keys()}

        for bn in concept_dict.keys():
            concept_bank_dct[bn] = ConceptBank(dict(bottleneck=bn, working_dir=path_to_working_dir,
                                                    concept_names=concept_dict[bn], class_id_dct=ace.class_to_id,
                                                    model_name=ace.model_name))

    del ace
    return concept_bank_dct
