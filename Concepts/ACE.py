"""ACE library.

Library for automatically discovering and testing concept activation vectors. It contains
ACE class that is able to discover the concepts belonging to one
of the possible classification labels of the classification task of a network
and calculate each concept's TCAV score.

This code is taken from Unsupervised discovery of concepts using the ACE method as described in
Ghorbani, A., Wexler, J., Zou, J., &#38; Kim, B. (2019). Towards automatic concept-based explanations.
Advances in Neural Information Processing Systems. https://github.com/amiratag/ACE..

The code is largely the same. Only minor changes, like not using tf.Session(), have been made.
"""
import shutil
import skimage.segmentation as segmentation
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional
from multiprocessing import dummy as multiprocessing
import tensorflow as tf
import sklearn.cluster as cluster
from sklearn.metrics.pairwise import euclidean_distances
import os
from Concepts.helper import load_images_from_files, get_activations_of_images, get_bottleneck_model
from Concepts.CAV import CAV, get_or_train_cav


class ACE:
    """ Unsupervised discovery of concepts using the ACE method as described in
        Ghorbani, A., Wexler, J., Zou, J., &#38; Kim, B. (2019). Towards automatic concept-based explanations.
        Advances in Neural Information Processing Systems. https://github.com/amiratag/ACE.

        ACE works in three stages.
        First, a segmentation method is used to segment the image in multi resolution image patches,
        For example, an image can be segmented in 15,50,80 different patches.

        Secondly, these patches are clustered together to be concepts. Outlier clusters are removed. These clusters
        represent concepts.

        Finally, using CAV and TCAVs these concepts are created and the corresponding TCAV scores are computed.
    """
    # TODO add work for grey scale images
    def __init__(self, model_name: str, bottlenecks: List, target_class: str, source_dir: str, working_dir: str,
                 random_concept: str, class_to_id: Dict, id_to_folder: Dict, average_image_value: int = 117, num_workers: int = 0,
                 channel_mean: bool = True, max_imgs: int = 40, min_imgs: int = 20, num_random_concepts: int = 20,
                 num_discovery_imgs=40) -> None:
        """Runs concept discovery algorithm. For more information see ACE docstring.

        @param model_name: Path to the tf.keras.Model or InceptionV3
        @param bottlenecks: A list of bottleneck layers for which ACE is performed.
        @param target_class: Name of the class for which the concepts need to be discovered.
        @param source_dir: Directory that contains folders of the images classes.
        @param random_concept: Name of the random_concept (used for statistical testing).
        @param class_to_id: Dictionary mapping the string representations to the integer representations
            for all classes.
        @param id_to_folder: Dictionary mapping the class IDs to the folder names
        @param average_image_value: An integer representing the average pixel value. Used for padding.
        @param num_workers: the number of worker threads to run in parallel.
        @param channel_mean: A boolean indicating whether to average out activations
            across multiple channels.
        @param max_imgs: Maximum number of patches in a discovered concept.
        @param min_imgs: minimum number of patches in a discovered concept for the concept to be accepted.
        @param num_discovery_imgs: Number of images used for concept discovery. If None, will use max_imgs instead.
        """
        self.model_name = model_name
        if model_name == 'InceptionV3':
            self.model = tf.keras.applications.inception_v3.InceptionV3()
        elif os.path.exists(self.model_name):
            self.model = tf.keras.models.load_model(self.model_name, compile=False)
        else:
            raise ValueError(f'{self.model_name} is not a directory to a model nor the InceptionV3model')
        self.bottlenecks = bottlenecks
        self.target_class = target_class
        self.working_dir = working_dir
        self.source_dir = source_dir
        self.random_concept = random_concept
        self.class_to_id = class_to_id
        self.id_to_folder = id_to_folder
        self.average_image_value = average_image_value  # 117 is default zero value for Inception V3
        self.image_shape = self.model.input.shape[1:3][::-1]  # retrieve image shape from the model as (width, height)
        self.num_workers = num_workers
        self.channel_mean = channel_mean
        self.max_imgs = max_imgs
        self.min_imgs = min_imgs
        self.num_random_concepts = num_random_concepts
        if num_discovery_imgs is None:
            num_discovery_imgs = max_imgs
        self.num_discovery_imgs = num_discovery_imgs
        self.concept_dict = None
        self.discovery_images = None
        self.dataset = None
        self.image_numbers = None
        self.patches = None
        self.activation_dir = os.path.join(working_dir, 'acts/')
        self.cav_dir = os.path.join(working_dir, 'cavs_temp/')
        self.discovered_concepts_dir = os.path.join(working_dir, 'concepts/')

    def load_concept_imgs(self, concept: str, max_imgs: int = 1000):
        """Loads all colored images of a concept or class. Rescales the images to self.image_shape if needed.

        @param str concept: The name of the concept to be loaded.
        @param int max_imgs: The amount of images of the concept or class to return.
        @return Images of the desired concept or class.
        """
        concept_dir = os.path.join(self.source_dir, self.id_to_folder[self.class_to_id[concept]])
        img_paths = [
            os.path.join(concept_dir, img)
            for img in os.listdir(concept_dir)
        ]
        return load_images_from_files(
            img_paths,
            max_imgs=max_imgs,
            return_filenames=False,
            do_shuffle=False,
            shape=self.image_shape,
            num_workers=self.num_workers)

    def create_patches_for_data(self, discovery_images: List = None, method: str = 'slic',
                                param_dict: Optional[Dict] = None) -> None:
        """Extracts patches for every image in discovery_images. For each image store all patches both upsized to
        original image size and the original patches.

        @param List discovery_images: List of all images used for creating the patches.
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
                lambda image: self.create_patches_of_img(image, method, param_dict), self.discovery_images)

            for idx, sp_outputs in enumerate(outputs):
                image_superpixels, image_patches = sp_outputs
                for superpixel, patch in zip(image_superpixels, image_patches):
                    dataset.append(superpixel)
                    patches.append(patch)
                    image_numbers.append(idx)
        else:
            # no parallelism used
            for idx, img in enumerate(self.discovery_images):
                image_superpixels, image_patches = self.create_patches_of_img(img, method, param_dict)
                for superpixel, patch in zip(image_superpixels, image_patches):
                    dataset.append(superpixel)
                    patches.append(patch)
                    image_numbers.append(idx)

        self.dataset, self.image_numbers, self.patches = np.array(dataset), np.array(image_numbers), np.array(patches)

    def create_patches_of_img(self, image: np.ndarray, method: str = 'slic',
                              param_dict: Optional[Dict] = None) -> Tuple[List, List]:
        """
        Extracts both the patches resized to original image size and the patches from a single image.
        Patches smaller than 1% of the image or patches too similar (jaccard > 0.5) to other patches are not included.

        @param np.ndarray image: Numpy array representing a single image. The image needs to be non-normalized.
        @param str method: String representing the segmentation method to be used.
            One of slic, watershed, quickshift, or felzenszwalb.
        @param Dict param_dict: Dictionary representing the parameters of the segmentation method. The keys are the
            parameter names and the values are lists representing the parameter values for each segmentation resolution.
        @return Tuple: Returns a tuple of all resized patches and original patches of a single image.
        @raise ValueError: if the chosen segmentation method is invalid.
        """
        # get the parameters of the chosen method
        if param_dict is None:
            param_dict = {}

        # get segmentation masks of different resolutions for the chosen methods
        unique_masks = []
        n_resolutions = param_dict.get('n_resolutions', 3)
        for i in range(n_resolutions):
            param_masks = []

            if method == 'slic':
                # get parameters of segmentation
                n_segments = param_dict.get('n_segments', [15, 50, 80])
                compactnesses = param_dict.get('compactness', [20] * n_resolutions)
                sigmas = param_dict.get('sigma', [1.] * n_resolutions)

                # perform segmentation
                segments = segmentation.slic(image, n_segments=n_segments[i], compactness=compactnesses[i],
                                             sigma=sigmas[i])

            elif method == 'watershed':
                # get parameters of segmentation
                markers = param_dict.get('marker', [15, 50, 80])
                compactnesses = param_dict.get('compactness', [0.] * n_resolutions)

                # perform segmentation
                segments = segmentation.watershed(image, markers=markers[i], compactness=compactnesses[i])

            elif method == 'quickshift':
                # get parameters of segmentation
                max_dists = param_dict.get('max_dist', [20, 15, 10])
                ratios = param_dict.get('ratio', [1.0] * n_resolutions)
                kernel_sizes = param_dict.get('kernel_size', [10] * n_resolutions)

                # perform segmentation
                segments = segmentation.quickshift(image, kernel_size=kernel_sizes[i], max_dist=max_dists[i],
                                                   ratio=ratios[i])

            elif method == 'felzenszwalb':
                # get parameters of segmentation
                scales = param_dict.get('scale', [1200, 500, 250])
                sigmas = param_dict.get('sigma', [0.8] * n_resolutions)
                min_sizes = param_dict.get('min_size', [20] * n_resolutions)

                # perform segmentation
                segments = segmentation.felzenszwalb(image, scale=scales[i], sigma=sigmas[i], min_size=min_sizes[i])

            else:
                raise ValueError('Invalid superpixel method!')

            # for each segmentation class find the unique segmentations (remove small or too similar masks)
            for s in range(segments.min(), segments.max() + 1):
                mask = (segments == s).astype(np.float32)
                # check if masks are not too similar to others or too small
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
            superpixel, patch = self._extract_patch(image, unique_masks.pop())
            superpixels.append(superpixel)
            patches.append(patch)
        return superpixels, patches

    def _extract_patch(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns a patch resized to original image size as well as the original patch of a single image.
        Pixels not in the patch are represented by average pixel values. The patch is up-sampled to the same size as the
        original image by applying bicubic resampling.

        @param np.ndarray image: The normalized image of which the patch should be extracted from.
        @param np.ndarray mask: The mask representing the patch which should be extracted.
        @return Tuple[np.ndarray, np.ndarray]: Returns a tuple consisting of a resized patch and the original patch.
        """
        mask_expanded = np.expand_dims(mask, -1)

        # change the zeros in the image to be the average value to differentiate padded pixels from black pixels
        patch = (mask_expanded * image + (1 - mask_expanded) * np.float32(self.average_image_value) / 255)

        # get the patch from the image
        ones = np.asarray(mask == 1).nonzero()
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))

        # resize the patch to be the same size as the original image using the BICUBIC interpolation method
        image_resized = np.asarray(image.resize(self.image_shape, Image.Resampling.BICUBIC)).astype(np.float32) / 255
        return image_resized, patch

    def _patch_activations(self, images: np.ndarray, bottleneck_layer: str) -> np.ndarray:
        """Returns activations of the bottleneck layer of an array of images.

        @param np.ndarray images: Array of images on which to calculate the activations of the bottleneck layer.
        @param str bottleneck_layer: Name of the bottleneck layer of the model of which the activations are calculated.
        @return np.ndarray: Array of the activations of the bottleneck layer for all the images.
        """
        activations = get_activations_of_images(images, get_bottleneck_model(self.model, bottleneck_layer))
        if self.channel_mean and len(activations.shape) > 3:
            output = np.mean(activations, (1, 2))  # average activations out across channels
        else:
            output = np.reshape(activations, [activations.shape[0], -1])
        return output

    def _cluster(self, activations, method='KM', param_dict=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs unsupervised clustering algorithm on concept activations.

        @param np.ndarray activations: Array containing the activation vectors of the bottleneck layer.
        @param str method: String representing the cluster method to be used. The following are supported:
            'KM': Kmeans Clustering
            'AP': Affinity Propagation
            'SC': Spectral Clustering
            'MS': Mean Shift clustering
            'DB': DBSCAN clustering method
        @param Dict param_dict: Contains superpixel method's parameters. If an empty dict is
            given, default parameters are used.
        @return cluster_labels np.ndarray: Cluster assignment label of each datapoint.
            cluster_costs np.ndarray: Clustering cost of each data point.
            centers np.ndarray: For methods like Affinity Propagation where they do not return a cluster center
                or a clustering cost, it calculates the medoid as the center and returns distance to center as each
                data points clustering cost.
        @raise ValueError: if the clustering method is invalid.
        """
        # Initialize and run Cluster algorithm
        if param_dict is None:
            param_dict = {}
        centers = None
        cluster_costs = None

        if method == 'KM':  # perform K-means clustering
            n_clusters = param_dict.pop('n_clusters', 25)
            km = cluster.KMeans(n_clusters)
            km.fit(activations)
            centers = km.cluster_centers_

            # compute distances from points to centers
            distances = np.linalg.norm(np.expand_dims(activations, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            cluster_labels, cluster_costs = np.argmin(distances, -1), np.min(distances, -1)

        elif method == 'AP':  # use Affinity Propagation
            damping = param_dict.pop('damping', 0.5)
            ca = cluster.AffinityPropagation(damping=damping)
            ca.fit(activations)
            centers = ca.cluster_centers_

            # compute distances from points to centers
            distances = np.linalg.norm(np.expand_dims(activations, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            cluster_labels, cluster_costs = np.argmin(distances, -1), np.min(distances, -1)

        elif method == 'MS':  # perform Mean Shift clustering
            ms = cluster.MeanShift(n_jobs=self.num_workers)
            cluster_labels = ms.fit_predict(activations)

        elif method == 'SC':  # perform Spectral clustering
            n_clusters = param_dict.pop('n_clusters', 25)
            sc = cluster.SpectralClustering(n_clusters=n_clusters, n_jobs=self.num_workers)
            cluster_labels = sc.fit_predict(activations)

        elif method == 'DB':  # perform DBSCAN
            eps = param_dict.pop('eps', 0.5)
            min_samples = param_dict.pop('min_samples', 20)
            sc = cluster.DBSCAN(eps, min_samples=min_samples, n_jobs=self.num_workers)
            cluster_labels = sc.fit_predict(activations)

        else:
            raise ValueError('Invalid Clustering Method!')

        # if clustering did not return cluster centers, use medoids as cluster center
        if centers is None:
            centers = np.zeros((np.max(cluster_labels) + 1, activations.shape[1]))
            cluster_costs = np.zeros(len(activations))
            for cluster_label in range(np.max(cluster_labels) + 1):
                cluster_idxs = np.where(cluster_labels == cluster_label)[0]
                cluster_points = activations[cluster_idxs]
                pairwise_distances = euclidean_distances(cluster_points)
                centers[cluster_label] = cluster_points[np.argmin(np.sum(pairwise_distances, -1))]
                cluster_costs[cluster_idxs] = np.linalg.norm(
                    activations[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
                    ord=2, axis=-1)

        return cluster_labels, cluster_costs, centers

    def discover_concepts(self,
                          method: str = 'KM',
                          activations: Optional[Dict] = None,
                          param_dicts: Optional[Dict] = None):
        """Discovers the frequent occurring concepts in the target class.

        Calculates self.dic, a dictionary containing all the information of the
        discovered concepts in the form of {'bottleneck layer name: bn_dic} where
        bn_dic itself is in the form of {'concepts:list of concepts,
        'concept name': concept_dic} where the concept_dic is in the form of
        {'images': resized patches of concept, 'patches': original patches of the
        concepts, 'image_numbers': image id of each patch}.

        @param method: Clustering method.
        @param activations: dictionary if activations are already calculated. If not calculates
            them. Must be a dictionary in the form of {'bn':array, ...}.
        @param param_dicts: A dictionary in the format of {'bottleneck':param_dict,...}
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

            # for each cluster compute how many segments belong to it relative to cluster size and amount of patches
            # subsequently remove outlier clusters
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
                            'image_numbers': self.image_numbers[concept_idxs]}
                        bn_dic[concept + '_center'] = centers[i]
            bn_dic.pop('label', None)
            bn_dic.pop('cost', None)
            self.concept_dict[bn] = bn_dic

    def _random_concept_activations(self, bottleneck_layer: str, bottleneck_model: tf.keras.Model, random_concept: str):
        """Wrapper for computing or loading activations of random concepts.

        Takes care of making, caching and loading activations.

        @param bottleneck_layer: The bottleneck layer name.
        @param bottleneck_model: The pretrained tensorflow model up to and including the bottleneck layer.
        @param str random_concept: The name of the random concept. e.g. "random500_0".
        @return A nested dict in the form of {concept:{bottleneck:activation}}.
        """

        if not os.path.exists(self.activation_dir):
            os.makedirs(self.activation_dir)
        rnd_acts_path = os.path.join(self.activation_dir, 'acts_{}_{}.npy'.format(
            random_concept, bottleneck_layer))
        if not os.path.exists(rnd_acts_path):
            rnd_imgs = self.load_concept_imgs(random_concept, self.max_imgs)
            activations = get_activations_of_images(rnd_imgs, bottleneck_model)
            np.save(rnd_acts_path, activations)
            del activations
            del rnd_imgs
        return np.load(rnd_acts_path).squeeze()

    def _calculate_cav(self, concept: str, random_concept: str, bottleneck: str, bottleneck_model: tf.keras.Model,
                       act_c: np.ndarray, ow: bool):
        """Calculates a single cav for a concept and one random counterpart.

          @param concept: Name of the concept
          @param random_concept: Name of the random counterpart.
          @param bottleneck: Name of the bottleneck layer.
          @param bottleneck_model: The pretrained tensorflow model up to and including the bottleneck layer.
          @param act_c: Activation matrix of the concept in the 'bottleneck' layer.
          @param ow: If True, overwrites already existing CAV.
          @return The accuracy of the computed CAV.
        """
        act_r = self._random_concept_activations(bottleneck, bottleneck_model, random_concept)
        act_r = act_r.reshape((act_r.shape[0], -1))  # flatten to allow for training
        cav_instance = get_or_train_cav([concept, random_concept], bottleneck,
                                        act_dct={concept: act_c, random_concept: act_r}, cav_dir=self.cav_dir, ow=ow)
        return cav_instance.accuracy

    def _concept_cavs(self, bottleneck: str, bottleneck_model: tf.keras.Model, concept: str,
                      activations: np.ndarray, randoms: Optional[List] = None, ow: bool = True) -> List:
        """Calculates CAVs of a concept versus all the random counterparts.

        @param bottleneck: Bottleneck layer name.
        @param bottleneck_model: The pretrained tensorflow model up to and including the bottleneck layer.
        @param concept: Concept name.
        @param activations: Activations of the concept in the bottleneck layer.
        @param randoms: None if the class random concepts are going to be used. Otherwise, a list with the names of the
            random concepts against which the CAVs will be trained
        @param ow: If True, overwrites existing CAVs.
        @return accuracies: A List of the accuracies of the concept versus all random counterparts.
        """
        activations = activations.reshape((activations.shape[0], -1))  # flatten activation matrix
        if randoms is None:
            randoms = ['random500_{}'.format(i) for i in np.arange(self.num_random_concepts)]  # load in random concepts

        # compute CAVs
        if self.num_workers:
            pool = multiprocessing.Pool(self.num_workers)
            accuracies = pool.map(
                lambda random: self._calculate_cav(concept, random, bottleneck, bottleneck_model, activations, ow),
                randoms)
        else:
            accuracies = []
            for random_concept in randoms:
                accuracies.append(self._calculate_cav(concept, random_concept, bottleneck, bottleneck_model,
                                                      activations, ow))
        return accuracies

    def cavs(self, min_acc: float = 0., ow: bool = False, in_memory: bool = True, concept_dir: str = '') -> Dict:
        """This method calculates and saves CAVs for all the discovered concepts
        versus all random concepts in all the bottleneck layers.

        @param min_acc: Delete discovered concept if average classification accuracy of the CAV is less than min_acc.
        @param ow: If True, overwrites already calculated CAVs.
        @param in_memory: If True, concept images are loaded from the self.concept_dict
            If False, concept images are loaded from the concept_dir.
        @param concept_dir: Directory where concept images are stored.
        @return acc: A dictionary of classification accuracy of linear boundaries orthogonal to CAV vectors of the form
         {'bottleneck layer':{'concept name':[list of accuracies], ...}, ...}. Also includes the random concept and the
         target_class as 'concept name'.
        """
        os.makedirs(self.cav_dir)
        accuracy = {bottleneck: {} for bottleneck in self.bottlenecks}
        concepts_to_delete = []
        for bottleneck in self.bottlenecks:
            # compute concept activations
            # prevent tf.function tracing by defining model outside for loop
            bottleneck_model = get_bottleneck_model(self.model, bottleneck)  # get keras model up to the bottleneck layer
            for concept in self.concept_dict[bottleneck]['concepts']:
                # load concept images
                if in_memory:
                    concept_imgs = self.concept_dict[bottleneck][concept]['images']
                else:
                    filepaths = [os.path.join(concept_dir, concept, bottleneck, img) for img in
                                 os.listdir(os.path.join(concept_dir, concept, bottleneck))]
                    concept_imgs = load_images_from_files(filepaths, do_shuffle=False)

                # get activations w.r.t the bottleneck layer of the concept images
                concept_acts = get_activations_of_images(concept_imgs, bottleneck_model)
                accuracy[bottleneck][concept] = self._concept_cavs(bottleneck, bottleneck_model, concept, concept_acts,
                                                                   ow=ow)  # compute cavs

                # delete if average accuracy of separating concept against random concepts is smaller
                # than user defined number
                if np.mean(accuracy[bottleneck][concept]) < min_acc:
                    concepts_to_delete.append((bottleneck, concept))

            # compute target_class activations
            target_class_acts = get_activations_of_images(self.discovery_images, bottleneck_model)
            accuracy[bottleneck][self.target_class] = self._concept_cavs(bottleneck, bottleneck_model,
                                                                         self.target_class, target_class_acts, ow=ow)
            # compute random activations
            rnd_acts = self._random_concept_activations(bottleneck, bottleneck_model, self.random_concept)
            accuracy[bottleneck][self.random_concept] = self._concept_cavs(bottleneck, bottleneck_model,
                                                                           self.random_concept, rnd_acts, ow=ow)

        # delete inaccurate concepts
        for bottleneck, concept in concepts_to_delete:
            self.delete_concept(bottleneck, concept)
        return accuracy

    def delete_concept(self, bottleneck_layer: str, concept: str):
        """Removes a discovered concepts if it's not already removed.

        @param bottleneck_layer: Bottleneck layer where the concepts are discovered.
        @param concept: Name of the concept to be removed.
        """
        self.concept_dict[bottleneck_layer].pop(concept, None)
        if concept in self.concept_dict[bottleneck_layer]['concepts']:
            self.concept_dict[bottleneck_layer]['concepts'].\
                pop(self.concept_dict[bottleneck_layer]['concepts'].index(concept))

    def aggregate_cavs(self, accuracies, mode='max'):
        randoms = ['random500_{}'.format(i) for i in range(self.num_random_concepts)]
        if mode == 'max':  # take maximum accuracy cav
            for bn in accuracies.keys():
                bn_dic = accuracies[bn]
                for concept in bn_dic:
                    max_random_concept = randoms[np.argmax(bn_dic[concept])]
                    filename = os.path.join(self.cav_dir, f'{bn}-{concept}-{max_random_concept}.pkl')
                    cav = CAV.load_cav(filename)
                    cav.file_name = f'{cav.bottleneck}-{cav.concept}.pkl'
                    cav.save_cav(os.path.join(self.working_dir, 'cavs'))

        elif mode == 'average':
            for bn in accuracies.keys():
                bn_dic = accuracies[bn]

                def load_cavs(random):
                    file = os.path.join(self.cav_dir, f'{bn}-{concept}-{random}.pkl')
                    cav_t = CAV.load_cav(file)
                    return cav_t.cav

                for concept in bn_dic:
                    if self.num_workers:
                        pool = multiprocessing.Pool(self.num_workers)
                        cavs = pool.map(lambda rnd: load_cavs(rnd), randoms)
                    else:
                        cavs = [load_cavs(random) for random in randoms]
                    cav = CAV.load_cav(os.path.join(self.cav_dir, f'{bn}-{concept}-random500_0.pkl'))
                    cav.cav = np.mean(np.array(cavs), axis=0)
                    cav.file_name = f'{cav.bottleneck}-{cav.concept}.pkl'
                    cav.save_cav(os.path.join(self.working_dir, 'cavs'))
        shutil.rmtree(self.cav_dir)
