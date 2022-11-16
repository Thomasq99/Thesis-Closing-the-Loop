"""ACE library.

Library for automatically discovering and testing concept activation vectors. It contains
ACE class that is able to discover the concepts belonging to one
of the possible classification labels of the classification task of a network
and calculate each concept's TCAV score.

This code is taken from Unsupervised discovery of concepts using the ACE method as described in
Ghorbani, A., Wexler, J., Zou, J., &#38; Kim, B. (2019). Towards automatic concept-based explanations.
Advances in Neural Information Processing Systems. https://github.com/amiratag/ACE..

The code is largely the same. Only minor changes, like not using tf.Session(), have been applied.
"""

import skimage.segmentation as segmentation
import numpy as np
import numpy.typing as npt
from PIL import Image
from typing import Tuple, List, Dict, Optional
from multiprocessing import dummy as multiprocessing
import tensorflow as tf
import sklearn.cluster as cluster
from sklearn.metrics.pairwise import euclidean_distances
import os
from preprocessing import load_images_from_files, get_activations_of_images, get_gradients_of_images
from tcav import cav
import scipy.stats as stats

class ACE:
    """ Unsupervised discovery of concepts using the ACE method as described in
        Ghorbani, A., Wexler, J., Zou, J., &#38; Kim, B. (2019). Towards automatic concept-based explanations.
        Advances in Neural Information Processing Systems. https://github.com/amiratag/ACE.

        Explanation .... TODO add explanation
    """
    # TODO add work for grey scale images
    def __init__(self, model: tf.keras.Model, bottlenecks: List, target_class: str, source_dir: str, activation_dir: str,
                 cav_dir: str, random_concept: str, class_to_id: Dict, average_image_value: int = 117, num_workers: int = 20,
                 channel_mean: bool = True, max_imgs: int = 40, min_imgs: int = 20, num_random_concepts: int = 2,
                 num_discovery_imgs=40) -> None:
        """TODO add explanation

        @param tf.Keras.Model model: A trained tensorflow model for which the concepts need to be discovered
        @param List bottlenecks: A list of bottleneck layers for which ACE is performed
        @param str target_class: Name of the class for which the concepts need to be discovered
        @param str source_dir: Directory that contains folders of the images classes
        @param str activation_dir: Directory to save computed activations
        @param str cav_dir: Directory to save computed CAVs
        @param str random_concept: Name of the random_concept (used for statistical testing)
        @param Dict class_to_id: Dictionary mapping the string representations to the integer representations
            for all classes
        @param int average_image_value: An integer representing the average pixel value. Used for padding
        @param int num_workers: the number of worker threads to run in parallel
        @param bool channel_mean: A boolean indicating whether or not to average out activations
            across multiple channels.
        @param int max_imgs: Maximum number of patches in a discovered concept
        @param int min_imgs: minimum number of patches in a discovered concept for the concept to be accepted
        @param: num_discovery_images: Number of images used for concept discovery. If None, will use max_imgs instead.
        """
        self.model = model
        self.bottlenecks = bottlenecks
        self.target_class = target_class
        self.activation_dir = activation_dir
        self.source_dir = source_dir
        self.cav_dir = cav_dir
        self.random_concept = random_concept
        self.class_to_id = class_to_id
        self.average_image_value = average_image_value # 117 is default zero value for Inception V3
        self.image_shape = model.layers[0].input_shape[1:3]  # retrieve image shape from the model
        self.num_workers = num_workers
        self.channel_mean = channel_mean
        self.max_imgs = max_imgs
        self.min_imgs = min_imgs
        self.num_random_concepts = num_random_concepts
        if num_discovery_imgs is None:
            num_discovery_imgs = max_imgs
        self.num_discovery_imgs = num_discovery_imgs

    def load_concept_imgs(self, concept, max_imgs = 1000):
        """Loads all colored images of a concept or class. Rescales the images to self.image_shape if needed.

        @param str concept: The name of the concept to be loaded
        @param int max_imgs: The amount of images of the concept or class to return
        @return Images of the desired concept or class
        """
        concept_dir = os.path.join(self.source_dir, concept)
        img_paths = [
            os.path.join(concept_dir, img)
            for img in os.listdir(concept_dir)
        ]
        return load_images_from_files(
            img_paths,
            max_imgs=max_imgs,
            return_filenames=False,
            do_shuffle=False,
            run_parallel=(self.num_workers > 0),
            shape=self.image_shape,
            num_workers=self.num_workers)

    def create_patches_for_data(self, discovery_images: List = None, method: str = 'slic',
                                param_dict: Optional[Dict] = None) -> None:
        """Extracts patches for every image in discovery_images. For each image store all patches both upsized to
        original image size and the original patches.

        @param List discovery_images: List of all images used for creating the patches
        @param str method: String representing the segmentation method to be used.
            One of slic, watershed, quickshift, or felzenszwalb.
        @param Dict param_dict: Dictionary representing the parameters of the segmentation method. The keys are the
            parameter names and the values are lists representing the parameter values for each segmentation resolution.
        """
        if param_dict is None:
            param_dict = {}
        dataset, image_numbers, patches = [], [], []

        if discovery_images is None:
            raw_imgs = self.load_concept_imgs(self.target_class, self.num_discovery_imgs)
            self.discovery_images = raw_imgs
        else:
            self.discovery_images = discovery_images
        # use multiple worker threads to extract patches in parallel
        if self.num_workers:
            pool = multiprocessing.Pool(self.num_workers)
            outputs = pool.map(
                lambda img: self.create_patches_of_img(img, method, param_dict),
                self.discovery_images)
            for idx, sp_outputs in enumerate(outputs):
                image_superpixels, image_patches = sp_outputs
                for superpixel, patch in zip(image_superpixels, image_patches):
                    dataset.append(superpixel)
                    patches.append(patch)
                    image_numbers.append(idx)
        else:
            # no parallelism used
            for idx, img in enumerate(self.discovery_images):
                print(img.shape)
                image_superpixels, image_patches = self.create_patches_of_img(
                    img, method, param_dict)
                for superpixel, patch in zip(image_superpixels, image_patches):
                    dataset.append(superpixel)
                    patches.append(patch)
                    image_numbers.append(idx)

        self.dataset, self.image_numbers, self.patches = np.array(dataset), np.array(image_numbers), np.array(patches)

    def create_patches_of_img(self, img: npt.ArrayLike, method: str = 'slic',
                              param_dict: Optional[Dict] = None) -> Tuple[List, List]:
        """
        Extracts both the patches resized to original image size and the patches from a single image.
        Patches smaller than 1% of the image or patches too similar (jaccard > 0.5) to other patches are not included.

        @param npt.ArrayLike img: Numpy array representing a single image. The image needs to be non-normalized.
        @param str method: String representing the segmentation method to be used.
            One of slic, watershed, quickshift, or felzenszwalb.
        @param Dict param_dict: Dictionary representing the parameters of the segmentation method. The keys are the
            parameter names and the values are lists representing the parameter values for each segmentation resolution.
        @return Tuple: Returns a tuple of all resized patches and original patches of a single image.
        @raise ValueError: if the chosen segmentation method is invalid
        """
        # Get the parameters of the chosen method
        if param_dict is None:
            param_dict = {}
        if method == 'slic':
            n_segments = param_dict.get('n_segments', [15, 50, 80])
            n_params = len(n_segments)
            compactnesses = param_dict.get('compactness', [20] * n_params)
            sigmas = param_dict.get('sigma', [1.] * n_params)
        elif method == 'watershed':
            markers = param_dict.get('marker', [15, 50, 80])
            n_params = len(markers)
            compactnesses = param_dict.get('compactness', [0.] * n_params)
        elif method == 'quickshift':
            max_dists = param_dict.get('max_dist', [20, 15, 10])
            n_params = len(max_dists)
            ratios = param_dict.get('ratio', [1.0] * n_params)
            kernel_sizes = param_dict.get('kernel_size', [10] * n_params)
        elif method == 'felzenszwalb':
            scales = param_dict.get('scale', [1200, 500, 250])
            n_params = len(scales)
            sigmas = param_dict.get('sigma', [0.8] * n_params)
            min_sizes = param_dict.get('min_size', [20] * n_params)
        else:
            raise ValueError('Invalid superpixel method!')

        # Get segmentation masks of different resolutions for the chosen methods
        unique_masks = []
        for i in range(n_params):
            param_masks = []
            if method == 'slic':
                segments = segmentation.slic(
                    img, n_segments=n_segments[i], compactness=compactnesses[i],
                    sigma=sigmas[i])
            elif method == 'watershed':
                segments = segmentation.watershed(
                    img, markers=markers[i], compactness=compactnesses[i])
            elif method == 'quickshift':
                segments = segmentation.quickshift(
                    img, kernel_size=kernel_sizes[i], max_dist=max_dists[i],
                    ratio=ratios[i])
            elif method == 'felzenszwalb':
                segments = segmentation.felzenszwalb(
                    img, scale=scales[i], sigma=sigmas[i], min_size=min_sizes[i])

            # for each segmentation class find the unique segmentations (remove small or too similar masks)
            for s in range(segments.min(), segments.max() + 1):
                mask = (segments == s).astype(float)
                # check if masks are not too similar or too small
                if np.mean(mask) > 0.001:  # if segment is larger than 1% of the image
                    unique = True
                    for seen_mask in unique_masks:  # compare  current mask to previously found masks
                        jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                        if jaccard > 0.5:  # do not store if too similar to previously found mask
                            unique = False
                            break
                    if unique:
                        param_masks.append(mask)
            # store all unique masks
            unique_masks.extend(param_masks)

        # resize patches to same size
        superpixels, patches = [], []
        while unique_masks:
            superpixel, patch = self._extract_patch(img, unique_masks.pop())
            superpixels.append(superpixel)
            patches.append(patch)
        return superpixels, patches

    def _extract_patch(self, image: npt.ArrayLike, mask: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """ Returns a patch resized to original image size as well as the original patch of a single image.
        Pixels not in the patch are represented by average pixel values. The patch is upsampled to the same size as the
        original image by applying bicubic resampling.

        @param npt.ArrayLike image: The normalized image of which the patch should be extracted from
        @param npt.ArrayLike mask: The mask representing the patch which should be extracted
        @return Tuple[npt.ArrayLike, npt.ArrayLike]: Returns a tuple consisting of a resized patch and the original patch
        """
        mask_expanded = np.expand_dims(mask, -1)
        #  change the zeros in the image to be the average value to differentiate padded pixels from black pixels
        patch = (mask_expanded * image + (
                1 - mask_expanded) * float(self.average_image_value) / 255)

        # Get the patch in the image
        ones = np.asarray(mask == 1).nonzero()
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
        # resize the patch to be the same size as the original image using the BICUBIC interpolation method
        image_resized = np.asarray(image.resize(self.image_shape,
                                              Image.Resampling.BICUBIC)).astype(float) / 255
        return image_resized, patch

    def _patch_activations(self, images: npt.ArrayLike, bottleneck_layer) -> npt.ArrayLike:
        """Returns activations of the bottleneck layer of an array of images.

        @param npt.ArrayLike images: Array of images on which to calculate the activations of the bottleneck layer.
        @param str bottleneck_layer: Name of the bottleneck layer of the model
            of which the activations are calculated.
        @return npt.ArrayLike: Array of the activations of the bottleneck layer for all the images.
        """
        activations = get_activations_of_images(images, self.model, bottleneck_layer)
        if self.channel_mean and len(activations.shape) > 3:
            output = np.mean(activations, (1, 2))  # average activations out across channels
        else:
            output = np.reshape(activations, [activations.shape[0], -1])
        return output

    def _cluster(self, activations, method='KM', param_dict=None) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Runs unsupervised clustering algorithm on concept activations.

        @param npt.ArrayLike activations: Array containing the activation vectors of the bottleneck layer
        @param str method: String representing the cluster method to be used. The following are supported:
            'KM': Kmeans Clustering
            'AP': Affinity Propagation
            'SC': Spectral Clustering
            'MS': Mean Shift clustering
            'DB': DBSCAN clustering method
        @param Dict param_dict: Contains superpixel method's parameters. If an empty dict is
            given, default parameters are used.
        @return cluster_labels npt.ArrayLike: Cluster assignment label of each datapoint
            cluster_costs npt.ArrayLike: Clustering cost of each data point
            centers npt.ArrayLike: For methods like Affinity Propagetion where they do not return a cluster center
                or a clustering cost, it calculates the medoid as the center and returns distance to center as each
                data points clustering cost.
        @raise ValueError: if the clustering method is invalid
        """
        if param_dict is None:
            param_dict = {}
        centers = None
        if method == 'KM':
            n_clusters = param_dict.pop('n_clusters', 25)
            km = cluster.KMeans(n_clusters)
            km.fit(activations)
            centers = km.cluster_centers_
            d = np.linalg.norm(
                np.expand_dims(activations, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            cluster_labels, cluster_costs = np.argmin(d, -1), np.min(d, -1)
        elif method == 'AP':
            damping = param_dict.pop('damping', 0.5)
            ca = cluster.AffinityPropagation(damping=damping)
            ca.fit(activations)
            centers = ca.cluster_centers_
            d = np.linalg.norm(
                np.expand_dims(activations, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            cluster_labels, cluster_costs = np.argmin(d, -1), np.min(d, -1)
        elif method == 'MS':
            ms = cluster.MeanShift(n_jobs=self.num_workers)
            cluster_labels = ms.fit_predict(activations)
        elif method == 'SC':
            n_clusters = param_dict.pop('n_clusters', 25)
            sc = cluster.SpectralClustering(
                n_clusters=n_clusters, n_jobs=self.num_workers)
            cluster_labels = sc.fit_predict(activations)
        elif method == 'DB':
            eps = param_dict.pop('eps', 0.5)
            min_samples = param_dict.pop('min_samples', 20)
            sc = cluster.DBSCAN(eps, min_samples, n_jobs=self.num_workers)
            cluster_labels = sc.fit_predict(activations)
        else:
            raise ValueError('Invalid Clustering Method!')
        if centers is None:  # If clustering did not return cluster centers, use medoids as cluster center
            centers = np.zeros((cluster_labels.max() + 1, activations.shape[1]))
            cluster_costs = np.zeros(len(activations))
            for cluster_label in range(cluster_labels.max() + 1):
                cluster_idxs = np.where(cluster_labels == cluster_label)[0]
                cluster_points = activations[cluster_idxs]
                pw_distances = euclidean_distances(cluster_points)
                centers[cluster_label] = cluster_points[np.argmin(np.sum(pw_distances, -1))]
                cluster_costs[cluster_idxs] = np.linalg.norm(
                    activations[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
                    ord=2, axis=-1)
        return cluster_labels, cluster_costs, centers

    def discover_concepts(self,
                          method='KM',
                          activations=None,
                          param_dicts=None):
        """Discovers the frequent occurring concepts in the target class.

        Calculates self.dic, a dicationary containing all the information of the
        discovered concepts in the form of {'bottleneck layer name: bn_dic} where
        bn_dic itself is in the form of {'concepts:list of concepts,
        'concept name': concept_dic} where the concept_dic is in the form of
        {'images': resized patches of concept, 'patches': original patches of the
        concepts, 'image_numbers': image id of each patch}

        @param str method: Clustering method.
        @param Dict activations: If activations are already calculated. If not calculates
            them. Must be a dictionary in the form of {'bn':array, ...}
        @param Dict param_dicts: A dictionary in the format of {'bottleneck':param_dict,...}
            where param_dict contains the clustering method's parameters
            in the form of {'param1':value, ...}. For instance for Kmeans
            {'n_clusters':25}. param_dicts can also be in the format
            of param_dict where same parameters are used for all
            bottlenecks.
        """
        if param_dicts is None:
            param_dicts = {}
        if set(param_dicts.keys()) != set(self.bottlenecks):
            param_dicts = {bn: param_dicts for bn in self.bottlenecks}
        self.concept_dict = {}  # The main dictionary storing the concepts of the ConceptDiscovery class.
        for bn in self.bottlenecks:
            bn_dic = {}
            # compute the activations of the patches for each bottleneck
            if activations is None or bn not in activations.keys():
                bn_activations = self._patch_activations(self.dataset, bn)
            else:
                bn_activations = activations[bn]
            # cluster the activations such that similar segments belong to the same cluster
            bn_dic['label'], bn_dic['cost'], centers = self._cluster(bn_activations, method, param_dicts[bn])
            concept_number, bn_dic['concepts'] = 0, []
            # For each cluster compute how many segments belong to it relative to cluster size and amount of patches
            # Remove outlier clusters
            for i in range(bn_dic['label'].max() + 1):
                label_idxs = np.where(bn_dic['label'] == i)[0]  # idx of patches belonging to cluster i
                if len(label_idxs) > self.min_imgs:
                    concept_costs = bn_dic['cost'][label_idxs]
                    concept_idxs = label_idxs[np.argsort(concept_costs)[:self.max_imgs]]
                    concept_image_numbers = set(self.image_numbers[label_idxs])
                    discovery_size = len(self.discovery_images)
                    highly_common_concept = len(concept_image_numbers) > 0.5 * len(label_idxs)
                    mildly_common_concept = len(concept_image_numbers) > 0.25 * len(label_idxs)
                    mildly_populated_concept = len(concept_image_numbers) > 0.25 * discovery_size
                    cond2 = mildly_populated_concept and mildly_common_concept
                    non_common_concept = len(concept_image_numbers) > 0.1 * len(label_idxs)
                    highly_populated_concept = len(concept_image_numbers) > 0.5 * discovery_size
                    cond3 = non_common_concept and highly_populated_concept
                    if highly_common_concept or cond2 or cond3:  # filter out unimportant concepts
                        concept_number += 1
                        concept = '{}_concept{}'.format(self.target_class, concept_number)
                        bn_dic['concepts'].append(concept)
                        bn_dic[concept] = {
                            'images': self.dataset[concept_idxs],
                            'patches': self.patches[concept_idxs],
                            'image_numbers': self.image_numbers[concept_idxs]
                        }
                        bn_dic[concept + '_center'] = centers[i]
            bn_dic.pop('label', None)
            bn_dic.pop('cost', None)
            self.concept_dict[bn] = bn_dic

    def _random_concept_activations(self, bottleneck_layer, random_concept):
        """Wrapper for computing or loading activations of random concepts.

        Takes care of making, caching (if desired) and loading activations.

        @param str bottleneck_layer: The bottleneck layer name
        @param str random_concept: The name of the random concept. e.g. "random500_0"
        @return A nested dict in the form of {concept:{bottleneck:activation}}
        """

        if not os.path.exists(self.activation_dir):
            os.makedirs(self.activation_dir)
        rnd_acts_path = os.path.join(self.activation_dir, 'acts_{}_{}.npy'.format(
            random_concept, bottleneck_layer))
        if not os.path.exists(rnd_acts_path):
            rnd_imgs = self.load_concept_imgs(random_concept, self.max_imgs)
            activations = get_activations_of_images(rnd_imgs, self.model, bottleneck_layer)
            np.save(rnd_acts_path, activations, allow_pickle=False)
            del activations
            del rnd_imgs
        return np.load(rnd_acts_path).squeeze()

    def _calculate_cav(self, concept, random_concept, bottleneck, act_c, ow):
        """Calculates a single cav for a concept and one random counterpart.

          @param concept: concept name
          @param random_concept: random concept name
          @param bottleneck: bottleneck layer name
          @param act_c: activation matrix of the concept in the 'bn' layer
          @param ow: overwrite if CAV already exists
          @return The accuracy of the computed CAV
        """
        act_r = self._random_concept_activations(bottleneck, random_concept)

        cav_instance = cav.get_or_train_cav([concept, random_concept], bottleneck,
                                            {concept: {bottleneck: act_c}, random_concept: {bottleneck: act_r}},
                                            cav_dir=self.cav_dir, overwrite=ow)
        return cav_instance.accuracies['overall']

    def _concept_cavs(self, bottleneck, concept, activations, randoms=None, ow=True):
        """Calculates CAVs of a concept versus all the random counterparts.

        @param bottleneck: Bottleneck layer name
        @param concept: Concept name
        @param activations: Activations of the concept in the bottleneck layer
        @param randoms: None if the class random concepts are going to be used
        @param ow: If True, overwrites existing CAVs
        @return accuracies: A List of the accuracies of the concept versus all random counterparts.
        """
        if randoms is None:
            randoms = ['random500_{}'.format(i) for i in np.arange(self.num_random_concepts)]
        if self.num_workers:
            pool = multiprocessing.Pool(self.num_workers)
            accuracies = pool.map(lambda rnd: self._calculate_cav(concept, rnd, bottleneck, activations, ow), randoms)
        else:
            accuracies = []
            for rnd in randoms:
                accuracies.append(self._calculate_cav(concept, rnd, bottleneck, activations, ow))
        return accuracies

    def cavs(self, min_acc=0., ow=True):
        """Calculates cavs for all discovered concepts.

        This method calculates and saves CAVs for all the discovered concepts
        versus all random concepts in all the bottleneck layers

        @param min_acc: Delete discovered concept if the average classification accuracy of the CAV is less than min_acc
        @param ow: If True, overwrites already calculated CAVs
        @return acc: A dictionary of classification accuracy of linear boundaries orthogonal to CAV vectors of the form
         {'bottleneck layer':{'concept name':[list of accuracies], ...}, ...}. Also includes the random concept and the
         target_class as 'concept name'.
        """
        acc = {bottleneck: {} for bottleneck in self.bottlenecks}
        concepts_to_delete = []
        for bottleneck in self.bottlenecks:
            # compute concept activations
            for concept in self.concept_dict[bottleneck]['concepts']:
                concept_imgs = self.concept_dict[bottleneck][concept]['images']
                concept_acts = get_activations_of_images(concept_imgs, self.model, bottleneck)
                acc[bottleneck][concept] = self._concept_cavs(bottleneck, concept, concept_acts, ow=ow)
                if np.mean(acc[bottleneck][concept]) < min_acc:
                    concepts_to_delete.append((bottleneck, concept))

            # compute target_class activations
            target_class_acts = get_activations_of_images(self.discovery_images, self.model, bottleneck)
            acc[bottleneck][self.target_class] = self._concept_cavs(bottleneck, self.target_class,
                                                                    target_class_acts, ow=ow)
            # compute random activations
            rnd_acts = self._random_concept_activations(bottleneck, self.random_concept)
            acc[bottleneck][self.random_concept] = self._concept_cavs(bottleneck, self.random_concept, rnd_acts, ow=ow)

        # delete inaccurate concepts
        for bottleneck, concept in concepts_to_delete:
            self.delete_concept(bottleneck, concept)
        return acc

    def delete_concept(self, bottleneck_layer, concept):
        """Removes a discovered concepts if it's not already removed.

        @param bottleneck_layer: Bottleneck layer where the concepts are discovered.
        @param concept: Name of the concept to be removed.

        Args:
          bn: Bottleneck layer where the concepts is discovered.
          concept: concept name
        """
        self.concept_dict[bottleneck_layer].pop(concept, None)
        if concept in self.concept_dict[bottleneck_layer]['concepts']:
            self.concept_dict[bottleneck_layer]['concepts'].\
                pop(self.concept_dict[bottleneck_layer]['concepts'].index(concept))

    def load_cav_direction(self, concept, random_concept, bottleneck_layer):
        """Loads an already computed cav.

        @param concept: the name of the concept which CAV has to be loaded.
        @param random_concept: The name of the random concept for which the CAV has been trained against.
        @param bottleneck_layer: The name of the bottleneck_layer for which the CAV has been computed.
        @return The CAV instance (normalized by vector norm2).
        """
        cav_key = cav.CAV.cav_key([concept, random_concept], bottleneck_layer, 'linear', 0.01)
        cav_path = os.path.join(self.cav_dir, cav_key.replace('/', '.') + '.pkl')
        vector = cav.CAV.load_cav(cav_path).cavs[0]
        return np.expand_dims(vector, 0) / np.linalg.norm(vector, ord=2)

    def _return_gradients(self, images):
        """For the given images calculates the gradient tensors. Represents the first term of the directional derivative
        in the TCAV paper. #TODO add reference

        @param images: Array of images for which we want to calculate the gradients
        @return A dictionary of the gradients per bottleneck. {bottleneck:gradients, ...:...}
        """

        gradients = {}
        class_id = self.class_to_id[self.target_class]
        for bottleneck_layer in self.bottlenecks:
            bottleneck_gradients = get_gradients_of_images(images, self.model, class_id, bottleneck_layer)
            gradients[bottleneck_layer] = bottleneck_gradients
        return gradients

    def _tcav_score(self, bottleneck_layer, concept, random_concept, gradients):
        """Calculates and returns the TCAV score of a concept.

        @param bottleneck_layer: Name of the bottleneck layer on which the TCAV score will be calculated.
        @param concept: Name of the concept of which the TCAV score will be calculated.
        @param random_concept: Name of the random concept on which the concept will be compared to.
        @param gradients: Dict of gradients of tcav_score_images.
        @return TCAV score of the concept w.r.t. the given random concept.
        """
        cav_ = self.load_cav_direction(concept, random_concept, bottleneck_layer)
        directional_derivative = np.sum(gradients[bottleneck_layer] * cav_, -1)
        return np.mean(directional_derivative > 0)
        #TODO why did they have smaller than 0? I believe because they used loss instead of logits
        #TODO there is discrepancy between the pytorch and tensorflow installation

    def tcavs(self, test: bool = False, sort: bool = True, tcav_score_images=None):
        """Calculates TCAV scores for all discovered concepts and sorts concepts.

        This method calculates TCAV scores of all the discovered concepts for
        the target class using all the calculated CAVs. It later sorts concepts
        based on their TCAV scores.

        @param test: If True, perform statistical testing and removes concepts that do not pass.
        @param sort: If True, sort the concepts in each bottleneck layer based on the average TCAV score of the concepts.
        @param tcav_score_images: Target class images used for calculating TCAV scores. If None the target class source
            directory images will be used.
        @return A dictionary of the form {'bottleneck layer':{'concept name': [list of tcav scores], ...}, ...}
        containing TCAV scores.
        """

        tcav_scores = {bn: {} for bn in self.bottlenecks}
        randoms = ['random500_{}'.format(i) for i in np.arange(self.num_random_concepts)]
        if tcav_score_images is None:  # Load target class images if not given
            raw_imgs = self.load_concept_imgs(self.target_class, 2 * self.max_imgs)
            tcav_score_images = raw_imgs[-self.max_imgs:]
        gradients = self._return_gradients(tcav_score_images)
        for bn in self.bottlenecks:
            for concept in self.dic[bn]['concepts'] + [self.random_concept]:
                def t_func(rnd):
                    return self._tcav_score(bn, concept, rnd, gradients)

                if self.num_workers:
                    pool = multiprocessing.Pool(self.num_workers)
                    tcav_scores[bn][concept] = pool.map(lambda rnd: t_func(rnd), randoms)
                else:
                    tcav_scores[bn][concept] = [t_func(rnd) for rnd in randoms]
        if test:
            self.test_and_remove_concepts(tcav_scores)
        if sort:
            self._sort_concepts(tcav_scores)
        return tcav_scores

    def do_statistical_testings(self, samples1: List, samples2: List):
        """Conducts t-test to compare two set of samples. In particular, if the means of the two samples are
        staistically different.

        @param samples1: First group of samples. In this use case, the TCAV scores of the concept.
        @param samples2: Second group of samples. In this use case, the TCAV scores of the random counterpart.
        @return p-value of the statistical test.
        """
        min_len = min(len(samples1), len(samples2))
        _, p = stats.ttest_rel(samples1[:min_len], samples2[:min_len])
        return p

    def test_and_remove_concepts(self, tcav_scores: Dict):
        """Performs statistical testing for all discovered concepts.

        Using TCAV scores of the discovered concepts versus the random_counterpart
        concept, performs statistical t-tests and removes concepts that have a p-value larger than 0.01

        @param tcav_scores: Dictionary containing the tcav scores in the form of {bottleneck:concept:[tcav_scores],...}.
        """
        concepts_to_delete = []
        for bottleneck in self.bottlenecks:
            for concept in self.concept_dict[bottleneck]['concepts']:
                pvalue = self.do_statistical_testings(tcav_scores[bottleneck][concept],
                                                      tcav_scores[bottleneck][self.random_concept])
                if pvalue > 0.01:
                    concepts_to_delete.append((bottleneck, concept))
        for bottleneck, concept in concepts_to_delete:
            self.delete_concept(bottleneck, concept)
