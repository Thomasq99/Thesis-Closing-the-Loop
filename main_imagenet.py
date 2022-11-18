import numpy as np

from ACE import ACE
from ACE_helper import *
import os
import shutil
#TODO error when multiprocessing and loading images

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # my GPU is too small to save enough images in its VRAM to get the gradients


def main(model, target_class, bottlenecks, imagenet_folder, source_dir, working_dir, random_concept='random_discovery',
         num_random_concepts=20):

    # related DIRs on CNS to store results #######
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

    class_to_id = ace_create_source_dir_imagenet(imagenet_folder, source_dir, target_class,
                                                 random_concept=random_concept, num_random_concepts=num_random_concepts)

    # run ACE
    ace = ACE(model, bottlenecks, target_class, source_dir,
              activations_dir, cavs_dir, random_concept, class_to_id,
              num_random_concepts=num_random_concepts, num_workers=0)

    # create patches
    print("Creating patches")
    ace.create_patches_for_data()
    image_dir = os.path.join(discovered_concepts_dir, 'images')
    os.makedirs(image_dir)
    save_images(image_dir, (ace.discovery_images * 255).astype(np.uint8))  # save images used for creating patches

    # # Discovering Concepts
    print('discovering concepts')
    ace.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
    del ace.dataset  # Free memory
    del ace.image_numbers
    del ace.patches

    # Save discovered concept images (resized and original sized)
    save_concepts(ace, discovered_concepts_dir)

    # Calculating CAVs and TCAV scores
    print('calculating CAVs and TCAVs')
    cav_accuracies = ace.cavs(min_acc=0.0)
    print('calculating TCAV')
    scores = ace.tcavs(test=False)
    save_ace_report(ace, cav_accuracies, scores, results_summaries_dir + f'ace_results_{target_class}.txt')

    print('Started plotting concepts')
    # Plot examples of discovered concepts
    for bottleneck in ace.bottlenecks:
        plot_concepts(ace, bottleneck, 10, address=results_dir)

model = tf.keras.applications.inception_v3.InceptionV3()
target_class = 'pelican'
bottlenecks = ['mixed8']
imagenet_folder = './data/ImageNet'
source_dir = './data/ACE_ImageNet'
working_dir = f'./ACE_output/ImageNet'

main(model, target_class, bottlenecks, imagenet_folder, source_dir, working_dir)
