from Concepts.CAV import CAV
from Concepts.helper import load_images_from_files, get_gradients_of_images, get_grad_model
import os
from typing import List
from plotly.subplots import make_subplots
from skimage.segmentation import mark_boundaries
import numpy as np
import plotly.graph_objects as go
from typing import Dict
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import tensorflow as tf


class ConceptBank:
    def __init__(self, concept_dct: Dict):
        self.bottleneck = concept_dct.get('bottleneck', None)
        self.concept_names = concept_dct.get('concept_names', None)
        self.working_dir = concept_dct.get('working_dir', None)
        self.class_id_dct = concept_dct.get('class_id_dct', None)
        self.model_name = concept_dct.get('model_name', None)
        self.in_memory = concept_dct.get('in_memory', False)
        self.tcav_scores = concept_dct.get('tcav_scores', None)
        self.concepts = concept_dct.get('tcav_scores', None)

    def add_concept(self, concepts: List):
        if self.in_memory:
            # TODO implement
            raise ValueError('Not implemented when CAVs are in memory')
        else:
            self.concept_names.extend(concepts)

    def remove_concept(self, concept_name):
        remove_idx = self.concept_names.index(concept_name)
        del self.concept_names[remove_idx]
        if self.in_memory:
            del self.concepts[remove_idx]
            if self.tcav_scores:
                del self.tcav_scores[remove_idx]

    def rename_concept(self, old_name, new_name):
        pass

    def sort_concepts(self, discovery_images=None):
        if (self.tcav_scores is None) or (len(self.tcav_scores)) != (len(self.concept_names)):
            self.compute_tcav_scores(discovery_images=discovery_images)

        sorted_idxs = np.argsort(self.tcav_scores)[::-1]
        self.tcav_scores = list(np.array(self.tcav_scores)[sorted_idxs])
        self.concept_names = list(np.array(self.concept_names)[sorted_idxs])
        if self.in_memory:
            self.concepts = list(np.array(self.concepts)[sorted_idxs])

    def compute_tcav_scores(self, discovery_images=None):
        concept_names_ACE = [concept for concept in self.concept_names if 'userDefined' not in concept]
        concept_names_user_defined = [concept for concept in self.concept_names if 'userDefined' in concept]
        concept_classes = np.array([concept.split('_')[0] for concept in concept_names_ACE])
        if discovery_images is None:
            discovery_images = {}
            for concept in set(concept_classes):
                image_dir = os.path.join(self.working_dir, 'concepts', concept, 'images')
                discovery_images[concept] = load_images_from_files([os.path.join(image_dir, file) for file in
                                                                    os.listdir(image_dir)], do_shuffle=True,
                                                                   max_imgs=40)

        if not self.in_memory:
            self.load_in_memory()
            self.in_memory = False

        if self.model_name == 'InceptionV3':
            model = tf.keras.applications.inception_v3.InceptionV3()
        elif os.path.exists(self.model_name):
            model = tf.keras.models.load_model(self.model_name)
        else:
            raise ValueError(f'{self.model_name} is not a directory to a model nor the InceptionV3model')

        tcav_scores = []
        concepts_ACE = [self.concepts[i] for i in range(len(self.concepts)) if 'userDefined'
                        not in self.concept_names[i]]
        for concept_class in set(concept_classes):
            # get gradients
            class_id = self.class_id_dct[concept_class]
            concepts_of_concept_class = np.array(concepts_ACE)[concept_classes == concept_class]
            discovery_images_of_class = discovery_images[concept_class]
            gradients = get_gradients_of_images(discovery_images_of_class, get_grad_model(model, self.bottleneck),
                                                class_id)

            # compute tcav scores
            tcav_scores.extend([concept.compute_tcav_score(gradients) for concept in concepts_of_concept_class])

        tcav_scores.extend([float('-inf') for concept in concept_names_user_defined])
        self.tcav_scores = tcav_scores

        if not self.in_memory:
            self.concepts = None

    def to_dict(self):
        return {'bottleneck': self.bottleneck, 'concept_names': self.concept_names, 'working_dir': self.working_dir,
                'class_id_dct': self.class_id_dct, 'model_name': self.model_name, 'in_memory': self.in_memory,
                'concepts': self.concepts, 'tcav_scores': self.tcav_scores}

    def load_in_memory(self):
        if not self.in_memory:
            self.concepts = [CAV.load_cav(os.path.join(self.working_dir, 'cavs', f'{self.bottleneck}-{concept}.pkl'))
                             for concept in self.concept_names]
            self.in_memory = True
        else:
            print('Already in memory.')

    def load_concept_imgs_ace(self, concept_names, num_images, shape):
        concept_dir = os.path.join(self.working_dir, 'concepts')
        concepts_dict = {'concepts': concept_names}

        images_locs = [os.path.join(concept_dir, concept.split("_")[0], self.bottleneck, concept)
                       for concept in concept_names]
        patches_locs = [os.path.join(concept_dir, concept.split("_")[0], self.bottleneck, concept + '_patches')
                        for concept in concept_names]

        for i in range(len(concept_names)):

            images, filenames = load_images_from_files(filenames=[os.path.join(images_locs[i], file)
                                                                  for file in os.listdir(images_locs[i])[:num_images]],
                                                       return_filenames=True, do_shuffle=False, shape=shape)

            patches = load_images_from_files(filenames=[os.path.join(patches_locs[i], file)
                                                        for file in os.listdir(patches_locs[i])[:num_images]],
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

    def plot_concepts(self, num_images=10, shape=(60, 60)):
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
        idx = -1 # ensure that idx_user_defined gets the first user_defined tcav_score
        for idx, concept in enumerate(concept_names_ACE):
            title = ['']*2*num_images
            if self.tcav_scores is not None:
                subtitle = f'{concept}: TCAV Score of {self.tcav_scores[idx]}, CAV accuracy of ' \
                           f'{self.concepts[idx].accuracy}'
            else:
                subtitle = f'{concept}: CAV accuracy of {self.concepts[idx].accuracy}'
            title[middle] = subtitle
            subplot_titles.extend(title)

        for idx_user_defined, concept in enumerate(concept_names_user_defined):
            title = ['']*num_images
            if self.tcav_scores is not None:
                subtitle = f'{concept}: TCAV Score of {self.tcav_scores[idx + idx_user_defined + 1]}, CAV accuracy of '\
                           f'{self.concepts[idx + idx_user_defined + 1].accuracy}'
            else:
                subtitle = f'{concept}: CAV accuracy of {self.concepts[idx + idx_user_defined + 1].accuracy}'
            title[middle] = subtitle
            subplot_titles.extend(title)

        fig = make_subplots(n_rows, num_images, horizontal_spacing=0.1/num_images, vertical_spacing=0.3/n_rows,
                            shared_xaxes=True, shared_yaxes=True, subplot_titles=subplot_titles)

        j = 1
        discovery_images_dct = {}
        for class_ in {name.split('_')[0] for name in concept_names_ACE}:
            image_dir = os.path.join(self.working_dir, 'concepts', class_, 'images')
            discovery_images_dct[class_] = load_images_from_files([os.path.join(image_dir, file)
                                                                   for file in os.listdir(image_dir)],
                                                                  shape=shape, do_shuffle=False)
        if concept_names_ACE:
            current_class = concept_names_ACE[0].split('_')[0]
            discovery_images = discovery_images_dct[current_class]
            for concept in concept_names_ACE:
                if concept.split('_')[0] != current_class:
                    current_class = concept.split('_')[0]
                    discovery_images = discovery_images_dct[current_class]
                # TODO add support for different modes
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

    def plot_concepts_plt(self, num_images, shape=(60, 60)):
        if not self.in_memory:
            self.load_in_memory()
            self.in_memory = False

        concepts_dct = self.load_concept_imgs(num_images, shape)
        num_concepts = len(self.concept_names)
        plt.rcParams['figure.figsize'] = num_images * 2.1, 4.3 * num_concepts
        fig = plt.figure(figsize=(num_images * 2, 4 * num_concepts))
        outer = gridspec.GridSpec(num_concepts, 1, wspace=0., hspace=0.3)

        discovery_images_dct = {}
        for class_ in {name.split('_')[0] for name in self.concept_names}:
            image_dir = os.path.join(self.working_dir, 'concepts', class_, 'images')
            discovery_images_dct[class_] = load_images_from_files([os.path.join(image_dir, file)
                                                                   for file in os.listdir(image_dir)],
                                                                  shape=shape, do_shuffle=False)
        current_class = self.concept_names[0].split('_')[0]
        discovery_images = discovery_images_dct[current_class]

        for n, concept in enumerate(self.concept_names):
            if concept.split('_')[0] != current_class:
                current_class = concept.split('_')[0]
                discovery_images = discovery_images_dct[current_class]
            inner = gridspec.GridSpecFromSubplotSpec(2, num_images, subplot_spec=outer[n], wspace=0, hspace=0.1)
            concept_images = concepts_dct[concept]['images']
            concept_patches = concepts_dct[concept]['patches']
            concept_image_numbers = concepts_dct[concept]['image_numbers']
            idxs = np.arange(len(concept_images))[:num_images]
            for i, idx in enumerate(idxs):
                ax = plt.Subplot(fig, inner[i])

                img = np.uint8(concept_images[idx] * 255)
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == num_images // 2:
                    ax.set_title(concept)
                ax.grid(False)
                fig.add_subplot(ax)
                ax = plt.Subplot(fig, inner[i + num_images])
                mask = 1 - (np.mean(concept_patches[idxs[i]] == float(117) / 255,
                                    -1) == 1)  # hard coded 117, default avg for inception
                image = discovery_images[concept_image_numbers[idx]]
                image = np.uint8(mark_boundaries(image, mask, color=(1, 1, 0), mode='thick') * 255)
                ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(str(concept_image_numbers[idx]))
                ax.grid(False)
                fig.add_subplot(ax)
        plt.suptitle(self.bottleneck)

        if not self.in_memory:
            self.concepts = None
        return fig
