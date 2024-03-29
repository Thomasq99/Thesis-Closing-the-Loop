from Concepts.CAV import CAV
from Concepts.helper import load_images_from_files, get_gradients_of_images, get_grad_model, get_activations_of_images,\
    get_bottleneck_model
import os
from typing import List, Dict, Optional, Tuple
from plotly.subplots import make_subplots
from skimage.segmentation import mark_boundaries
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import scipy.stats as stats
from decimal import Decimal

class ConceptBank:
    """ Class to use as a concept bank. For a current session all concepts can be stored in a concept bank.
    Moreover, the TCAV scores of concepts in the concept bank can be computed. The concepts can also be visualized.

    """
    def __init__(self, concept_dct: Dict):
        """ Initializes a concept bank from a dictionary with format {concept_attr: value}"""
        self.bottleneck = concept_dct.get('bottleneck', None)
        self.concept_names = concept_dct.get('concept_names', None)
        self.working_dir = concept_dct.get('working_dir', None)
        self.class_id_dct = concept_dct.get('class_id_dct', None)
        self.model_name = concept_dct.get('model_name', None)
        self.in_memory = concept_dct.get('in_memory', False)
        self.tcav_scores = concept_dct.get('tcav_scores', None)
        self.concepts = concept_dct.get('concepts', None)
        self.p_vals = concept_dct.get('p_vals', None)

    def load_model(self):
        # load in tensorflow model
        if self.model_name == 'InceptionV3':
            model = tf.keras.applications.inception_v3.InceptionV3()
        elif os.path.exists(self.model_name):
            model = tf.keras.models.load_model(self.model_name)
        else:
            raise ValueError(f'{self.model_name} is not a directory to a model nor the InceptionV3model')

        return model

    def add_concept(self, concepts: List):
        """ Adds a list of concept(s) to the concept bank.

        @param concepts: List of concepts to add. [concept_name, concept_name2]"""
        if self.in_memory:
            # TODO implement
            raise ValueError('Not implemented when CAVs are in memory')
        else:
            self.concept_names.extend(concepts)

    def remove_concept(self, concept_name: str):
        """ Removes a concept from the Concept Bank.

        @param concept_name: Name of the concept to be removed"""
        remove_idx = self.concept_names.index(concept_name)
        del self.concept_names[remove_idx]
        if self.in_memory:
            del self.concepts[remove_idx]
        if self.tcav_scores:
            del self.tcav_scores[remove_idx]
            del self.p_vals[remove_idx]

    def sort_concepts(self, discovery_images=None):
        """Sorts concepts based on the TCAV scores of the concepts."""

        # compute TCAV scores if needed
        if (self.tcav_scores is None) or (len(self.tcav_scores)) != (len(self.concept_names)):
            self.compute_tcav_scores(discovery_images=discovery_images)

        sorted_idxs = np.argsort(self.tcav_scores)[::-1]

        # sort based on indices
        self.tcav_scores = list(np.array(self.tcav_scores)[sorted_idxs])
        self.concept_names = list(np.array(self.concept_names)[sorted_idxs])
        self.p_vals = list(np.array(self.p_vals)[sorted_idxs])
        if self.in_memory:
            self.concepts = list(np.array(self.concepts)[sorted_idxs])

    def compute_tcav_scores(self, discovery_images: Optional[Dict] = None):
        """Computes the TCAV scores of all concept in the Concept Bank. In particular, if discovery_images is not
        supplied images from the classes from which the concepts are extracted are used to compute the TCAV_score on.

        @param discovery_images: Dictionary containing an array of discovery images for each class.
            format is {concept_class_name: img_array}
        """
        model = self.load_model()
        if self.tcav_scores is not None:
            concept_names_to_compute = self.concept_names[-(len(self.concept_names) - len(self.tcav_scores)):]
        else:
            self.tcav_scores = []
            self.p_vals = []
            concept_names_to_compute = self.concept_names
        # For the discovered concepts get the classes for which they were discovered
        concept_classes = np.array([concept.split('__')[0] for concept in concept_names_to_compute])

        # load images if discovery_images is not supplied
        if discovery_images is None:
            discovery_images = {}
            for class_ in set(concept_classes):  # for each concept load the class images
                image_dir = os.path.join(self.working_dir, 'concepts', class_, 'images')
                discovery_images[class_] = load_images_from_files([os.path.join(image_dir, file) for file in
                                                                    os.listdir(image_dir)], do_shuffle=True,
                                                                   max_imgs=40, shape=model.input.shape[1:3][::-1])

        if not self.in_memory:
            self.load_in_memory()
            self.in_memory = False

        concepts_to_compute = [concept for concept in self.concepts if concept.concept in concept_names_to_compute]

        folder_path = os.path.join(self.working_dir, 'cavs_temp')
        cavs_random = [CAV.load_cav(os.path.join(folder_path, filename)) for filename in os.listdir(folder_path)
                       if ('random_discovery' in filename) and (self.bottleneck in filename) and
                       ('random500' in filename)]

        # For each concept compute the TCAV scores.
        tcav_scores = self.tcav_scores
        p_vals = self.p_vals
        for concept_class in set(concept_classes):
            # get gradients of images w.r.t. bottleneck layer
            class_id = self.class_id_dct[concept_class]
            concepts_of_concept_class = np.array(concepts_to_compute)[concept_classes == concept_class]
            discovery_images_of_class = discovery_images[concept_class]
            gradients = get_gradients_of_images(discovery_images_of_class, get_grad_model(model, self.bottleneck),
                                                class_id)
            # compute tcav scores random
            tcav_scores_random = [cav.compute_tcav_score(gradients) for cav in cavs_random]

            # compute tcav scores concepts
            for concept in concepts_of_concept_class:
                cavs = [CAV.load_cav(os.path.join(folder_path, filename)) for filename in os.listdir(folder_path)
                        if (str(concept.concept) + '-' in filename) and (self.bottleneck in filename)]
                tcav_scores_p_val = [cav.compute_tcav_score(gradients) for cav in cavs]
                p_vals.extend([stats.ttest_rel(tcav_scores_p_val, tcav_scores_random)[1]])
                tcav_scores.extend([concept.compute_tcav_score(gradients)])

        self.p_vals = p_vals
        self.tcav_scores = tcav_scores

        if not self.in_memory:
            # remove from memory
            self.concepts = None

    def to_dict(self) -> Dict:
        """ Exports a ConceptBank as a dictionary."""
        return {'bottleneck': self.bottleneck, 'concept_names': self.concept_names, 'working_dir': self.working_dir,
                'class_id_dct': self.class_id_dct, 'model_name': self.model_name, 'in_memory': self.in_memory,
                'concepts': self.concepts, 'tcav_scores': self.tcav_scores, 'p_vals': self.p_vals}

    def load_in_memory(self):
        """ Loads the CAVs of the concepts in memory"""
        if not self.in_memory:
            self.concepts = [CAV.load_cav(os.path.join(self.working_dir, 'cavs', f'{self.bottleneck}-{concept}.pkl'))
                             for concept in self.concept_names]
            self.in_memory = True
        else:
            print('Already in memory.')

    def load_concept_imgs_ace(self, concept_names: List, num_images: int, shape: Tuple[int, int]):
        """ Loads the images, patches and image numbers for each concept extracted using ACE.

        @param concept_names: List of the names of the concepts extracted using ACE
        @param num_images: Amount of images to load for each concept
        @param shape: Tuple denoting the shape of the images to be loaded as
        """
        concept_dir = os.path.join(self.working_dir, 'concepts')
        concepts_dict = {'concepts': concept_names}

        # get paths to images and patches
        image_paths = [os.path.join(concept_dir, concept.split("__")[0], self.bottleneck, concept)
                       for concept in concept_names]
        patch_paths = [os.path.join(concept_dir, concept.split("__")[0], self.bottleneck, concept + '_patches')
                        for concept in concept_names]

        for i in range(len(concept_names)):

            images, filenames = load_images_from_files(filenames=[os.path.join(image_paths[i], file)
                                                                  for file in os.listdir(image_paths[i])[:num_images]],
                                                       return_filenames=True, do_shuffle=False, shape=shape)

            patches = load_images_from_files(filenames=[os.path.join(patch_paths[i], file)
                                                        for file in os.listdir(patch_paths[i])[:num_images]],
                                             do_shuffle=False, shape=shape)

            image_numbers = [int(filename.split('_')[-1].split('.')[0]) for filename in filenames]
            concepts_dict[concept_names[i]] = {'images': images, 'patches': patches, 'image_numbers': image_numbers}

        return concepts_dict

    def load_concept_imgs_userDefined(self, concept_names, num_images, shape):
        concept_dir = os.path.join(self.working_dir, 'concepts')
        concepts_dict = {'concepts': concept_names}
        images_locs = [os.path.join(concept_dir, concept, 'images')
                       for concept in concept_names]

        for i in range(len(concept_names)):
            images = load_images_from_files(filenames=[os.path.join(images_locs[i], file)
                                                       for file in os.listdir(images_locs[i])[:num_images]],
                                            do_shuffle=False, shape=shape)
            concepts_dict[concept_names[i]] = images
        return concepts_dict

    def project_onto_conceptspace(self, images):
        model = self.load_model()
        if not self.in_memory:
            self.load_in_memory()
            self.in_memory = False

        activations = get_activations_of_images(images, get_bottleneck_model(model, self.bottleneck))
        activations = activations.reshape((activations.shape[0], -1))
        concepts = []
        concept_norms = []
        for concept in self.concepts:
            concepts.append(concept.cav)
            concept_norms.append(concept.norm)
        concept_matrix = np.concatenate(concepts, axis=0)
        concept_norms = np.array(concept_norms)
        projection = (activations @ concept_matrix.T)/(concept_norms**2)

        if not self.in_memory:
            # remove from memory
            self.concepts = None

        return projection

    def plot_concepts(self, num_images=10, shape=(60, 60), max_rows=None):
        if not self.in_memory:
            self.load_in_memory()
            self.in_memory = False

        concept_names_ACE = [concept for concept in self.concept_names if 'userDefined' not in concept]
        concept_names_user_defined = [concept for concept in self.concept_names if 'userDefined' in concept]

        concepts_dct_ACE = self.load_concept_imgs_ace(concept_names_ACE, num_images, shape)
        concepts_dct_user_defined = self.load_concept_imgs_userDefined(concept_names_user_defined, num_images, shape)

        n_rows = (2 * len(concept_names_ACE)) + len(concept_names_user_defined)

        # create subplot titles
        subplot_titles = []
        middle = num_images // 2
        idx = -1  # ensure that idx_user_defined gets the first user_defined tcav_score
        for idx, concept in enumerate(concept_names_ACE):
            title = ['']*2*num_images
            if self.tcav_scores is not None:
                subtitle = f'{concept}: TCAV Score: {self.tcav_scores[idx]}, p:{Decimal(self.p_vals[idx]):.2E}, CAV accuracy:' \
                           f'{self.concepts[idx].accuracy}'
            else:
                subtitle = f'{concept}: CAV accuracy:{self.concepts[idx].accuracy}'
            title[middle] = subtitle
            subplot_titles.extend(title)

        for idx_user_defined, concept in enumerate(concept_names_user_defined):
            title = ['']*num_images
            if self.tcav_scores is not None:
                subtitle = f'{concept}: TCAV Score:{self.tcav_scores[idx + idx_user_defined + 1]}, ' \
                           f'p:{Decimal(self.p_vals[idx + idx_user_defined + 1]):.2E},' \
                           f' CAV accuracy of {self.concepts[idx + idx_user_defined + 1].accuracy}'
            else:
                subtitle = f'{concept}: CAV accuracy of {self.concepts[idx + idx_user_defined + 1].accuracy}'
            title[middle] = subtitle
            subplot_titles.extend(title)

        if max_rows is not None:
            fig = make_subplots(max_rows, num_images, horizontal_spacing=0.1/num_images, vertical_spacing=0.3/max_rows,
                                shared_xaxes=True, shared_yaxes=True, subplot_titles=subplot_titles)
        else:
            fig = make_subplots(n_rows, num_images, horizontal_spacing=0.1/num_images, vertical_spacing=0.3/n_rows,
                                shared_xaxes=True, shared_yaxes=True, subplot_titles=subplot_titles)

        j = 1
        discovery_images_dct = {}
        for class_ in {name.split('__')[0] for name in concept_names_ACE}:
            image_dir = os.path.join(self.working_dir, 'concepts', class_, 'images')
            discovery_images_dct[class_] = load_images_from_files([os.path.join(image_dir, file)
                                                                   for file in os.listdir(image_dir)],
                                                                  shape=shape, do_shuffle=False)
        if concept_names_ACE:
            current_class = concept_names_ACE[0].split('__')[0]
            discovery_images = discovery_images_dct[current_class]
            for concept in concept_names_ACE:
                if concept.split('_')[0] != current_class:
                    current_class = concept.split('__')[0]
                    discovery_images = discovery_images_dct[current_class]
                concept_images = concepts_dct_ACE[concept]['images']
                concept_patches = concepts_dct_ACE[concept]['patches']
                concept_image_numbers = concepts_dct_ACE[concept]['image_numbers']
                idxs = np.arange(len(concept_images))[:num_images]
                for i, idx in enumerate(idxs):
                    image = np.uint8(concept_images[idx]*255)
                    mask = 1 - (np.mean(concept_patches[idxs[i]] == float(117) / 255, -1) == 1)  # 117, default avg for inception
                    annotated_image = discovery_images[concept_image_numbers[idx]]
                    annotated_image = np.uint8(mark_boundaries(annotated_image, mask, color=(1, 1, 0), mode='thick')*255)
                    fig.add_trace(go.Image(z=image, hoverinfo='none'), j, i+1)
                    fig.add_trace(go.Image(z=annotated_image, hoverinfo='none'), j + 1, i+1)
                j += 2

        for concept in concept_names_user_defined:
            concept_images = concepts_dct_user_defined[concept]
            idxs = np.arange(len(concept_images))[:num_images]
            for i, idx in enumerate(idxs):
                image = np.uint8(concept_images[idx]*255)
                fig.add_trace(go.Image(z=image, hoverinfo='none'), j, i + 1)
            j += 1

        # if max_rows is set
        if max_rows is not None:
            for q in range(max_rows - j):
                for i in range(num_images):
                    fig.add_trace(go.Image(z=np.zeros(shape=(1, 1)), hoverinfo='none'), q+j, i+1)

            fig.update_layout(autosize=True,
                              width=9 * 100,
                              height=max_rows * 100,
                              margin=dict(l=20, r=20, b=20, t=50),
                              overwrite=True)
        else:
            fig.update_layout(autosize=True,
                              width=9 * 100,
                              height=n_rows * 100,
                              margin=dict(l=20, r=20, b=20, t=50),
                              overwrite=True)
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

        if not self.in_memory:
            self.concepts = None

        return fig

    def plot_concept(self, concept_name, num_images, shape):

        if 'userDefined' not in concept_name:
            fig = make_subplots(2, num_images, shared_xaxes=True, shared_yaxes=True)
            concepts_dct = self.load_concept_imgs_ace([concept_name], num_images, shape=shape)
            class_ = concept_name.split('__')[0]
            image_dir = os.path.join(self.working_dir, 'concepts', class_, 'images')
            discovery_images = load_images_from_files([os.path.join(image_dir, file) for file in os.listdir(image_dir)],
                                                      shape=shape, do_shuffle=False)

            concept_images = concepts_dct[concept_name]['images']
            concept_patches = concepts_dct[concept_name]['patches']
            concept_image_numbers = concepts_dct[concept_name]['image_numbers']
            idxs = np.arange(len(concept_images))[:num_images]
            for i, idx in enumerate(idxs):
                image = np.uint8(concept_images[idx] * 255)
                mask = 1 - (np.mean(concept_patches[idxs[i]] == float(117) / 255, -1) == 1)  # 117, default avg for inception
                annotated_image = discovery_images[concept_image_numbers[idx]]
                annotated_image = np.uint8(mark_boundaries(annotated_image, mask, color=(1, 1, 0), mode='thick') * 255)
                fig.add_trace(go.Image(z=image, hoverinfo='none'), 1, i + 1)
                fig.add_trace(go.Image(z=annotated_image, hoverinfo='none'), 2, i + 1)

            fig.update_layout(autosize=True,
                              width=(num_images-1) * 200,
                              height=2 * 200,
                              margin=dict(l=20, r=20, b=20, t=50),
                              overwrite=True)
        else:
            fig = make_subplots(1, num_images, shared_xaxes=True, shared_yaxes=True)
            concepts_dct = self.load_concept_imgs_userDefined([concept_name], num_images, shape=shape)
            concept_images = concepts_dct[concept_name]
            idxs = np.arange(len(concept_images))[:num_images]
            for i, idx in enumerate(idxs):
                image = np.uint8(concept_images[idx]*255)
                fig.add_trace(go.Image(z=image, hoverinfo='none'), 1, i + 1)

            fig.update_layout(autosize=True,
                              width=(num_images-1) * 200,
                              height=1 * 200,
                              margin=dict(l=20, r=20, b=20, t=50),
                              overwrite=True)

        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)


        return fig
