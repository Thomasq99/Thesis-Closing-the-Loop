import os
import numpy as np
from PIL import Image
import multiprocessing.dummy as multiprocessing
import tensorflow as tf
import matplotlib.gridspec as gridspec
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import json
from typing import Dict, Tuple, List, Union
import scipy.stats as stats


def ace_create_source_dir_from_array(images: np.ndarray, labels: np.ndarray, target_class: str,
                                     source_dir: str, class_to_id: Dict, max_imgs_target_class: int = 100,
                                     max_imgs_random: int = 500, num_random_concepts: int = 20,
                                     ow: bool = False) -> None:
    """Helper function to transform data from an array to the required data format for running ACE. In particular,
    a folder of data will be created in source_dir. Images for the target_class and random concepts are stored in
    subfolders in this folder.

    @param images: Array containing the images. Form of (n, width, height, 3).
    @param labels: Array containing the labels of the images.
    @param target_class: Name of the target class for which concepts will be discovered.
    @param source_dir: Name of the directory where the formatted data for ACE will be stored.
    @param class_to_id: Dictionary mapping label names to class ids.
    @param max_imgs_target_class: Maximum amount of images to be stored for the target class.
    @param max_imgs_random: Maximum amount of images to be stored for the random concepts
    @param num_random_concepts: Number of random concepts to be created.
    @param ow: If True, existing folders, like existing random concepts, will be overwritten.
    """

    output_dir_lst = [f'{source_dir}/{target_class}'] + [f'{source_dir}/random_discovery'] +\
                     [f'{source_dir}/random{max_imgs_random}_{i}' for i in range(num_random_concepts)]

    for output_dir in output_dir_lst:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            no_of_imgs = 0
        elif ow:  # overwrite current images
            no_of_imgs = 0
        else:  # use previous images
            no_of_imgs = len(os.listdir(output_dir))

        # create target_class image folder
        if output_dir.endswith(target_class) and no_of_imgs < max_imgs_target_class:
            target_class_idxs = np.argwhere(labels == class_to_id[target_class])[:, 0]
            create_concept(output_dir, images[target_class_idxs], max_imgs_target_class, no_of_imgs)
        # create random concept folders
        elif not output_dir.endswith(target_class) and no_of_imgs < max_imgs_random:
            create_concept(output_dir, images, max_imgs_random, no_of_imgs)
        else:
            print(f'{output_dir} was already created and has enough images')


def create_concept(output_dir: str, images: np.ndarray, max_imgs: int, no_of_imgs: int) -> None:
    """ Function that saves max_imgs of images to be used for a concept in the output_dir.

    @param output_dir: Name of the target directory to store the data.
    @param images: Array of images to sample from and save in output_dir.
    @param max_imgs: Maximum number of images that can be stored in output_dir.
    @param no_of_imgs: Number of images currently in output_dir.
    """
    imgs_needed = max_imgs - no_of_imgs
    if imgs_needed > images.shape[0]:
        imgs_needed = images.shape[0]

    # sample and save images in output_dir
    np.random.shuffle(images)
    for idx, img in enumerate(images[:imgs_needed, :]):
        image = Image.fromarray(img)
        image.save(f'{output_dir}/img_{idx + no_of_imgs}.png')


def ace_create_source_dir_imagenet(imagenet_folder_path: str, source_dir: str, target_class: str,
                                   target_shape: Tuple = (299, 299),  num_random_concepts: int = 20,
                                   max_imgs_target_class: int = 100, ow: bool = False) -> Dict:
    """Helper function to transform data from the imagenet_folder_path directory to the required data format
    for running ACE. In particular, a folder of data will be created in source_dir. Images for the target_class and
    random concepts are stored in subfolders in this folder.

    @param imagenet_folder_path: Path to the ImageNet directory.
    @param source_dir: Path to the directory where the data will be stored.
    @param target_class: Name of the class for which the concepts will be discovered.
    @param target_shape: Shape in which the images will be stored (width, height).
    @param num_random_concepts: Number of random concepts that will be created.
    @param max_imgs_target_class: Number of images to store for the target_class.
    @param ow: If True, existing data folders will be overwritten.
    @return: Dictionary mapping the names of the classes to the class ids.
    """
    # create class_to_id mapping
    with open(os.path.join(imagenet_folder_path, 'imagenet_class_index.json')) as jsonFile:
        imagenet_dct = json.load(jsonFile)
        jsonFile.close()

    class_to_id = {}
    id_to_folder = {}
    for key, value in imagenet_dct.items():
        id_to_folder[int(key)] = value[0]
        class_to_id[value[1]] = int(key)

    output_dir_lst = [f'{source_dir}/{target_class}'] + [f'{source_dir}/random_discovery'] + \
                     [f'{source_dir}/random500_{i}' for i in range(num_random_concepts)]

    for output_dir in output_dir_lst:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            no_of_imgs = 0
        elif ow:  # overwrite current images
            no_of_imgs = 0
        else:  # use previous images
            no_of_imgs = len(os.listdir(output_dir))

        # create target_class folder
        if output_dir.endswith(target_class) and no_of_imgs < max_imgs_target_class:
            folder_name = id_to_folder[class_to_id[target_class]]
            target_class_loc = None
            for folder in ['train.X1', 'train.X2', 'train.X3', 'train.X4']:
                if folder_name in os.listdir(os.path.join(imagenet_folder_path, folder)):
                    target_class_loc = os.path.join(imagenet_folder_path, folder, folder_name)

            if target_class_loc is None:
                raise ValueError('Invalid target_class')

            # get sample of images to move to source_dir
            images_names_to_copy = np.random.choice(np.array(os.listdir(target_class_loc)), max_imgs_target_class,
                                                    replace=False)
            # move images to source_dir
            for image_name in images_names_to_copy:
                img = Image.open(os.path.join(target_class_loc, image_name))
                img = img.resize(target_shape).convert('RGB')  # resize images to desired shape
                img.save(os.path.join(output_dir, f'img_{no_of_imgs}.png'))
                no_of_imgs += 1
                if no_of_imgs >= 500:
                    break
            print(f'{output_dir} created')

        # create random_concepts
        elif not output_dir.endswith(target_class) and no_of_imgs < 500:
            for i in range(no_of_imgs, 500):
                train_folder = np.random.choice(np.array(['train.X1', 'train.X2', 'train.X3', 'train.X4']))
                class_folder = os.path.join(os.path.join(imagenet_folder_path, train_folder),
                                            np.random.choice(np.array(os.listdir(os.path.join(imagenet_folder_path,
                                                                                              train_folder)))))
                image_path = os.path.join(class_folder, np.random.choice(np.array(os.listdir(class_folder))))
                img = Image.open(image_path)
                img = img.resize(target_shape).convert('RGB')
                img.save(os.path.join(output_dir, f'img_{i}.png'))
            print(f'{output_dir} created')
        else:
            print(f'{output_dir} was already created and already has enough images')
    return class_to_id


def load_image_from_file(filename: str, shape: Tuple):
    """Given a filename, try to open the file. If failed, return None.

    @param filename: Name of the image file to load.
    @param shape: Desired shape of the image (width, height).
    @return: The image if succeeds, None if fails.
    @raise: Exception if the image was not the right shape.
    """
    if not os.path.exists(filename):
        print('Cannot find file: {}'.format(filename))
        return None
    try:
        #  get and, if desired, reshape image
        img = Image.open(filename)
        if img.size != shape:
            img = img.resize(shape, Image.BILINEAR)
        img = np.array(img)
        # normalize pixel values to between 0 and 1
        img = np.float32(img) / 255.0
        # TODO add functionality for greyscale
        if not (len(img.shape) == 3 and img.shape[2] == 3):  # only works for RGB images with channel last
            return None
        else:
            return img

    except Exception as e:
        print(e)
        return None


def load_images_from_files(filenames: List, max_imgs: int = 500, return_filenames: bool = False,
                           do_shuffle: bool = True, shape: Tuple = (299, 299),
                           num_workers: int = 100) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """Return image arrays from filenames.

    @param filenames: List of paths to image files.
    @param max_imgs: Maximum number of images to load from the filenames.
    @param return_filenames: If True, return Array of images and filenames.
    @param do_shuffle: If True, shuffle filenames before getting images.
    @param shape: Desired shape of the images (width, height).
    @param num_workers: number of workers to use in parallelization.
    @return Returns the loaded images. If return_filenames is True, also returns the filenames.
    """
    imgs = []
    if do_shuffle:
        np.random.shuffle(filenames)
    if return_filenames:
        final_filenames = []

    # load images from filenames using parallelization
    if num_workers:
        pool = multiprocessing.Pool(num_workers)
        imgs = pool.map(lambda filename: load_image_from_file(filename, shape), filenames[:max_imgs])
        if return_filenames:
            final_filenames = [f for i, f in enumerate(filenames[:max_imgs]) if imgs[i] is not None]
            imgs = [img for img in imgs if img is not None]
    # load images from filenames iteratively.
    else:
        for filename in filenames[:max_imgs]:
            img = load_image_from_file(filename, shape)
            if img is not None:
                imgs.append(img)
            if return_filenames:
                final_filenames.append(filename)
    if return_filenames:
        return np.array(imgs, dtype=object).astype(np.float32), final_filenames
    else:
        return np.array(imgs, dtype=object).astype(np.float32)


def get_bottleneck_model(model: tf.keras.Model, bottleneck_layer: str) -> tf.keras.Model:
    """Returns a model up to and including the bottleneck layer. Output of this model is the output of the bottleneck.

    @param model: Trained tf.keras.Model.
    @param bottleneck_layer: Name of the bottleneck layer.
    @return A model up to and including the bottleneck layer.
    """
    return tf.keras.Model(inputs=model.input, outputs=model.get_layer(bottleneck_layer).output)

def get_activations_of_images(images: np.ndarray, bottleneck_model: tf.keras.Model) -> np.ndarray:
    """Returns an Array of activations of the bottleneck layer of the model for all images.

    #TODO find out whether shape is corrrect
    @param images: Array of the form (n, w, h, c), where n is the number of images, w is the width of an image,
     h is the height of an image, and c is the number of channels. Usually 3 since ACE only works on RGB images.
    @param bottleneck_model: trained tf.keras.Model where the output is the output of the bottleneck model.
    @return: Array denoting the activations of the images.
    """

    activations = bottleneck_model.predict(images, batch_size=32).squeeze()
    return activations


def get_grad_model(model: tf.keras.Model, bottleneck_layer: str) -> tf.keras.Model:
    """ Returns a model with as output the bottleneck activations as well as the logits.

    @param model: Trained tf.keras.Model for which the gradients need to be calculated.
    @param bottleneck_layer: Name of the bottleneck layer for which we want the gradient.
    @return: Model that will output the bottleneck activations as well as the logits.
    """
    return tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(bottleneck_layer).output, model.output])


def get_gradients_of_images(images: np.ndarray, grad_model: tf.keras.Model, class_id: int) -> np.ndarray:
    """Compute the gradient of the logit of the class_id w.r.t. the images' activations of the bottleneck_layer in a
    pretrained model.

    @param images: Array of images of which to compute the gradient.
    @param grad_model: Model that outputs the activations of the bottleneck_layer and the logits.
    @param class_id: ID locating the logits of the class for which the gradients need to be computed.
    @return: gradients.
    """
    # get model up to bottleneck layer and up to the final layer
    with tf.GradientTape() as tape:
        # get activations and logits
        bottleneck_layers_out, predictions = grad_model(images)
        logit = predictions[:, class_id]
        # take gradients of logit w.r.t. the bottleneck_layer
        grads = tape.gradient(logit, bottleneck_layers_out)
        # flatten gradients to get gradient vector per image
        grads = tf.reshape(grads, shape=(images.shape[0], -1))
    return grads


def save_images(addresses: Union[List, str], images: List):
    """Save images in the addresses.

    @param addresses: Either list of addresses to save the images as or the name of the directory to save all images in.
    @param images: The list of all images in np.uint8 format.
    """
    # get paths for storing the images
    if not isinstance(addresses, list):
        image_addresses = []
        for i, image in enumerate(images):
            image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
            image_addresses.append(os.path.join(addresses, image_name))
        addresses = image_addresses
    assert len(addresses) == len(images), 'Invalid number of addresses'

    # save images to addresses
    for address, image in zip(addresses, images):
        with open(address, 'wb') as f:
            Image.fromarray(image).save(f, format='PNG')


def do_statistical_testings(samples1: List, samples2: List):
    """Conducts t-test to compare two set of samples. In particular, if the means of the two samples are
    staistically different.

    @param samples1: First group of samples. In this use case, the TCAV scores of the concept.
    @param samples2: Second group of samples. In this use case, the TCAV scores of the random counterpart.
    @return p-value of the statistical test.
    """
    min_len = min(len(samples1), len(samples2))
    _, p = stats.ttest_rel(samples1[:min_len], samples2[:min_len])
    return p

def save_concepts(cd, concepts_dir):
    """Saves discovered concepts' images and patches.

    @param cd: The ConceptDiscovery instance containing the concepts we want to save.
    @param concepts_dir: The directory to save the concepts' images.
    """
    for bn in cd.bottlenecks:
        for concept in cd.concept_dict[bn]['concepts']:
            patches_dir = os.path.join(concepts_dir, bn + '_' + concept + '_patches')
            images_dir = os.path.join(concepts_dir, bn + '_' + concept)
            patches = (np.clip(cd.concept_dict[bn][concept]['patches'], 0, 1) * 255).astype(np.uint8)
            images = (np.clip(cd.concept_dict[bn][concept]['images'], 0, 1) * 255).astype(np.uint8)
            os.makedirs(patches_dir)
            os.makedirs(images_dir)
            image_numbers = cd.concept_dict[bn][concept]['image_numbers']
            image_addresses, patch_addresses = [], []
            for i in range(len(images)):
                image_name = '0' * int(np.ceil(2 - np.log10(i + 1))) + '{}_{}'.format(i + 1, image_numbers[i])
                patch_addresses.append(os.path.join(patches_dir, image_name + '.png'))
                image_addresses.append(os.path.join(images_dir, image_name + '.png'))
            save_images(patch_addresses, patches)
            save_images(image_addresses, images)


def save_ace_report(cd, accs, scores, address):
    """Saves TCAV scores.

    Saves the average CAV accuracies and average TCAV scores of the concepts
    discovered in ConceptDiscovery instance.

    @param cd: The ConceptDiscovery instance.
    @param accs: The cav accuracy dictionary returned by cavs method of the ConceptDiscovery instance.
    @param scores: The tcav score dictionary returned by tcavs method of the ConceptDiscovery instance.
    @param address: The address to save the text file in.
    """
    report = '\n\n\t\t\t ---CAV accuracies---'
    for bn in cd.bottlenecks:
        report += '\n'
        for concept in cd.concept_dict[bn]['concepts']:
            report += '\n' + bn + ':' + concept + ':' + str(np.mean(accs[bn][concept]))
    with open(address, 'w') as f:
        f.write(report)
    report = '\n\n\t\t\t ---TCAV scores---'
    for bn in cd.bottlenecks:
        report += '\n'
        for concept in cd.concept_dict[bn]['concepts']:
            pvalue = cd.do_statistical_testings(scores[bn][concept], scores[bn][cd.random_concept])
            report += '\n{}:{}:{},{}'.format(bn, concept, np.mean(scores[bn][concept]), pvalue)
        report += '\n{}:{}:{},{}'.format(bn, cd.random_concept, np.mean(scores[bn][cd.random_concept]), 'na')
    with open(address, 'a') as f:
        f.write(report)


def plot_concepts(cd, bottleneck, target_class, num=10, address=None, mode='diverse', concepts=None):
    """Plots examples of discovered concepts.

    @param cd: The concept discovery instance.
    @param bottleneck: Bottleneck layer name.
    @param target_class: Name of the class of the concept.
    @param num: Number of images to print out of each concept.
    @param address: If not None, saves the output to the address as a .PNG image.
    @param mode: If 'diverse', it prints one example of each of the target class images
        is coming from. If 'random', randomly samples examples of the concept. If
        'max', prints out the most activating examples of that concept.
    @param concepts: If None, prints out examples of all discovered concepts.
        Otherwise, it should be either a list of concepts to print out examples of or just one concept's name
    @raise ValueError: if the mode is invalid.
    """
    if concepts is None:
        concepts = cd.concept_dict[bottleneck]['concepts']
    elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
        concepts = [concepts]
    num_concepts = len(concepts)
    plt.rcParams['figure.figsize'] = num * 2.1, 4.3 * num_concepts
    fig = plt.figure(figsize=(num * 2, 4 * num_concepts))
    outer = gridspec.GridSpec(num_concepts, 1, wspace=0., hspace=0.3)
    for n, concept in enumerate(concepts):
        inner = gridspec.GridSpecFromSubplotSpec(2, num, subplot_spec=outer[n], wspace=0, hspace=0.1)
        concept_images = cd.concept_dict[bottleneck][concept]['images']
        concept_patches = cd.concept_dict[bottleneck][concept]['patches']
        concept_image_numbers = cd.concept_dict[bottleneck][concept]['image_numbers']
        if mode == 'max':
            idxs = np.arange(len(concept_images))
        elif mode == 'random':
            idxs = np.random.permutation(np.arange(len(concept_images)))
        elif mode == 'diverse':
            idxs = []
            while len(idxs) < len(concept_images):
                seen = set()
                for idx in range(len(concept_images)):
                    if concept_image_numbers[idx] not in seen and idx not in idxs:
                        seen.add(concept_image_numbers[idx])
                        idxs.append(idx)
        else:
            raise ValueError('Invalid mode!')
        idxs = idxs[:num]
        for i, idx in enumerate(idxs):
            ax = plt.Subplot(fig, inner[i])
            ax.imshow(concept_images[idx])
            ax.set_xticks([])
            ax.set_yticks([])
            if i == int(num / 2):
                ax.set_title(concept)
            ax.grid(False)
            fig.add_subplot(ax)
            ax = plt.Subplot(fig, inner[i + num])
            mask = 1 - (np.mean(concept_patches[idx] == float(cd.average_image_value) / 255, -1) == 1)
            image = cd.discovery_images[concept_image_numbers[idx]]
            ax.imshow(mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(concept_image_numbers[idx]))
            ax.grid(False)
            fig.add_subplot(ax)
    plt.suptitle(bottleneck)
    if address is not None:
        fig.savefig(f'{address}{target_class}_{bottleneck}.png')
        plt.clf()
        plt.close(fig)
