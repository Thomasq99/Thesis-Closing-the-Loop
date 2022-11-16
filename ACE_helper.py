import os
import numpy as np
from PIL import Image
import multiprocessing.dummy as multiprocessing
import tensorflow as tf
import numpy.typing as npt
import matplotlib.gridspec as gridspec
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt


def ace_create_source_dir_from_array(images, label, target_class, source_dir, class_to_id, max_imgs_target_class=100,
                                     max_imgs_random=500, num_random_concepts=2, ow=True) -> None:
    """

    @param images:
    @param label:
    @param target_class:
    @param source_dir:
    @param class_to_id:
    @param max_imgs_target_class:
    @param max_imgs_random:
    @param num_random_concepts:
    @param ow:
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

        if output_dir.endswith(target_class) and no_of_imgs < max_imgs_target_class:
            target_class_idxs = np.argwhere(label == class_to_id[target_class])[:, 0]
            create_concept(output_dir, images[target_class_idxs], max_imgs_target_class, no_of_imgs)
        elif not output_dir.endswith(target_class) and no_of_imgs < max_imgs_random:
            create_concept(output_dir, images, max_imgs_random, no_of_imgs)
        else:
            print(f'{output_dir} was already created and already has enough images')


def create_concept(output_dir, images, max_imgs, no_of_imgs) -> None:
    """

    @param output_dir:
    @param images:
    @param max_imgs:
    @param no_of_imgs:
    @param label:
    """
    imgs_needed = max_imgs - no_of_imgs

    if imgs_needed > images.shape[0]:
        imgs_needed = images.shape[0]

    np.random.shuffle(images)
    for idx, img in enumerate(images[:imgs_needed, :]):
        image = Image.fromarray(img)
        image.save(f'{output_dir}/img_{idx + no_of_imgs}.jpg')


def load_image_from_file(filename, shape):
    """Given a filename, try to open the file. If failed, return None.
    Args:
    filename: location of the image file
    shape: the shape of the image file to be scaled
    Returns:
    the image if succeeds, None if fails.
    Rasies:
    exception if the image was not the right shape.
    @param filename:
    @param shape:
    @return: The image if succeeds, None if fails
    @raise: Exception if the image was not the right shape
    """
    if not os.path.exists(filename):
        print('Cannot find file: {}'.format(filename))
        return None
    try:
        img = np.array(Image.open(filename).resize(
            shape, Image.BILINEAR))
        # Normalize pixel values to between 0 and 1.
        img = np.float32(img) / 255.0
        # TODO add functionality for greyscale
        if not (len(img.shape) == 3 and img.shape[2] == 3): # only works for RGB images with channel last
            return None
        else:
            return img

    except Exception as e:
        print(e)
        return None


def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=True, run_parallel=True,
                           shape=(64, 64),
                           num_workers=100):
    """Return image arrays from filenames.
    Args:
    filenames: locations of image files.
    max_imgs: maximum number of images from filenames.
    return_filenames: return the succeeded filenames or not
    do_shuffle: before getting max_imgs files, shuffle the names or not
    run_parallel: get images in parallel or not
    shape: desired shape of the image
    num_workers: number of workers in parallelization.
    Returns:
    image arrays and succeeded filenames if return_filenames=True.
    """
    imgs = []
    # First shuffle a copy of the filenames.
    filenames = filenames[:]
    if do_shuffle:
        np.random.shuffle(filenames)
    if return_filenames:
        final_filenames = []
    if run_parallel:
        pool = multiprocessing.Pool(num_workers)
        imgs = pool.map(lambda filename: load_image_from_file(filename, shape), filenames[:max_imgs])
        if return_filenames:
            final_filenames = [f for i, f in enumerate(filenames[:max_imgs]) if imgs[i] is not None]
            imgs = [img for img in imgs if img is not None]
    else:
        for filename in filenames[:max_imgs]:
            img = load_image_from_file(filename, shape)
            if img is not None:
                imgs.append(img)
            if return_filenames:
                final_filenames.append(filename)
    if return_filenames:
        return np.array(imgs), final_filenames
    else:
        return np.array(imgs)


def get_activations_of_images(images: npt.ArrayLike, model: tf.keras.Model, bottleneck_layer: str) -> npt.ArrayLike:
    """Returns an Array of activations of the bottleneck layer of the model for all images.

    @param images: Array of the form (n, w, h, c), where n is the number of images, w is the width of an image,
     h is the height of an image, and c is the number of channels. Usually 3 since ACE only works on RGB images.
    @param model: trained tf.keras.Model used to get the activations.
    @param bottleneck_layer: str depicting the bottlneck layer for which the activations need to be computed.
    @return: Array denoting the activations of the images.
    """
    bottleneck_model = tf.keras.Model(inputs=model.input,
                                      outputs=model.get_layer(bottleneck_layer).output)
    activations = bottleneck_model.predict(images, batch_size=128).squeeze()
    return activations


def get_gradients_of_images(images: npt.ArrayLike, model: tf.keras.Model, class_id: int,
                            bottleneck_layer: str) -> npt.ArrayLike:
    """Compute the gradient of the logit of the class_id w.r.t. the images' activations of the bottleneck_layer in a
    pretrained model.

    @param images: Array of images of which to compute the gradient
    @param model: Model for which the gradients need to be computed
    @param class_id: ID locating the logits of the class for which the gradients need to be computed.
    @param bottleneck_layer: Name of the bottleneck_layer.
    @return: gradients
    """

    #TODO https://github.com/tensorflow/tcav/issues/124 shows that taking the gradients w.r.t. the logits gives a different answer
    #TODO might need to change this to loss instead of logits and compare.
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(bottleneck_layer).output, model.output])
    with tf.GradientTape() as tape:
        bottleneck_layers_out, predictions = grad_model(images)
        logit = predictions[:, class_id]
        # take gradients of last layer with respect to the bottleneck_layer
        grads = tape.gradient(logit, bottleneck_layers_out)
        # flatten gradients to get gradient vector per image
        grads = tf.reshape(grads, shape=(images.shape[0], -1))
    return grads


def save_images(addresses, images):
    """Save images in the addresses.

    @param addresses: Either list of addresses to save the images as or the name of the directory to save all images in.
    @param images: The list of all images in np.uint8 format.
    """
    if not isinstance(addresses, list):
        image_addresses = []
        for i, image in enumerate(images):
            image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
            image_addresses.append(os.path.join(addresses, image_name))
        addresses = image_addresses
    assert len(addresses) == len(images), 'Invalid number of addresses'
    for address, image in zip(addresses, images):
        with open(address, 'wb') as f:
            Image.fromarray(image).save(f, format='PNG')


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
    @param accs: The cav accuracy dictionary returned by cavs method of the ConceptDiscovery instance
    @param scores: The tcav score dictionary returned by tcavs method of the ConceptDiscovery instance
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
            report += '\n{}:{}:{},{}'.format(bn, concept,np.mean(scores[bn][concept]), pvalue)
    with open(address, 'a') as f:
        f.write(report)


def plot_concepts(cd, bottleneck, num=10, address=None, mode='diverse', concepts=None):
    """Plots examples of discovered concepts.

    @param cd: The concept discovery instance
    @param bottleneck: Bottleneck layer name
    @param num: Number of images to print out of each concept
    @param address: If not None, saves the output to the address as a .PNG image
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
            while True:
                seen = set()
                for idx in range(len(concept_images)):
                    if concept_image_numbers[idx] not in seen and idx not in idxs:
                        seen.add(concept_image_numbers[idx])
                        idxs.append(idx)
                    if len(idxs) == len(concept_images):
                        break
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
        with open(address + bottleneck + '.png', 'w') as f:
            fig.savefig(f)
        plt.clf()
        plt.close(fig)