import pickle as p
from Concepts.CAV import CAV
from Concepts.ACE_helper import load_images_from_files
import os
from typing import List
from plotly.subplots import make_subplots
from skimage.segmentation import mark_boundaries
import numpy as np
import plotly.graph_objects as go
from PIL import  Image


class ConceptBank:
    def __init__(self, bottleneck, working_dir, concept_names=None):
        if concept_names is None:
            concept_names = []
        self.bottleneck = bottleneck
        self.concept_names = concept_names
        self.working_dir = working_dir
        self.in_memory = False
        self.tcav_scores = []
        self.concepts = None

    def add_concept(self, concepts: List):
        self.concept_names.extend(concepts)

    def remove_concept(self, concept):
        pass

    def rename_concept(self, old_name, new_name):
        pass

    def sort_concepts(self):
        pass

    def to_dict(self):
        return {'bottleneck': self.bottleneck, 'concept_names': self.concept_names, 'working_dir':self.working_dir}

    def from_dict(self, dct):
        self.bottleneck = dct['bottleneck']
        self.concept_names = dct['concept_names']
        self.working_dir = dct['working_dir']

    def load_in_memory(self):
        if not self.in_memory:
            self.concepts = [CAV.load_cav(os.path.join(self.working_dir, 'cavs', concept))
                             for concept in self.concept_names]
            self.in_memory = True
        else:
            print('Already in memory.')

    def load_concept_imgs(self, num_images=10):
        concept_dir = os.path.join(self.working_dir, 'concepts')
        concepts_dict = {'concepts': self.concept_names}


        images_locs = [os.path.join(concept_dir, concept.split("_")[0], self.bottleneck, concept)
                       for concept in self.concept_names]
        patches_locs = [os.path.join(concept_dir, concept.split("_")[0], self.bottleneck, concept + '_patches')
                        for concept in self.concept_names]

        for i in range(len(self.concept_names)):

            images, filenames = load_images_from_files(filenames=[os.path.join(images_locs[i], file)
                                                                  for file in os.listdir(images_locs[i])[:num_images]],
                                                       return_filenames=True, do_shuffle=False)

            patches = load_images_from_files(filenames=[os.path.join(patches_locs[i], file)
                                                        for file in os.listdir(patches_locs[i])[:num_images]],
                                             do_shuffle=False)

            image_numbers = [int(filename.split('_')[-1].split('.')[0]) for filename in filenames]
            concepts_dict[self.concept_names[i]] = {'images': images, 'patches': patches, 'image_numbers': image_numbers}

        return concepts_dict

    def plot_concepts(self, num_images=10):
        concepts_dct = self.load_concept_imgs(num_images)

        n_rows = 2 * len(self.concept_names)
        n_cols = 10

        # create subplot titles
        subplot_titles = []
        for concept in self.concept_names:
            # add subplot title in the middle every other row
            middle = n_cols // 2
            title = [''] * 2 * n_cols
            title[middle] = concept  # TODO add average accuracy
            subplot_titles.extend(title)

        fig = make_subplots(n_rows, n_cols, horizontal_spacing=0.01, vertical_spacing=0.01,
                            column_widths=[1 / n_cols] * n_cols, shared_xaxes=True, shared_yaxes=True,
                            subplot_titles=subplot_titles)

        j = 1
        current_class = self.concept_names[0].split('_')[0]
        image_dir = os.path.join(self.working_dir, 'concepts', current_class, 'images')
        discovery_images = load_images_from_files([os.path.join(image_dir, file) for file in os.listdir(image_dir)],
                                                  do_shuffle=False)

        for concept in self.concept_names:
            # TODO add support for different modes, add bottleneck chooser, optional add functionality for number of images
            concept_images = concepts_dct[concept]['images']
            concept_patches = concepts_dct[concept]['patches']
            concept_image_numbers = concepts_dct[concept]['image_numbers']
            idxs = np.arange(len(concept_images))[:n_cols]
            for i, idx in enumerate(idxs):
                img = Image.fromarray(np.uint8(concept_images[idx] * 255)).resize((60, 60))  # reduce image size
                fig.add_trace(go.Image(z=img), j, i + 1)
                mask = 1 - (np.mean(concept_patches[idxs[i]] == float(117) / 255, -1) == 1)  # hard coded 117, default avg for inception
                image = discovery_images[concept_image_numbers[idx]]
                image = Image.fromarray(np.uint8(mark_boundaries(image, mask, color=(1, 1, 0),
                                                                 mode='thick') * 255)).resize((60, 60))
                fig.add_trace(go.Image(z=image), j + 1, i + 1)
            j += 2

            if concept.split('_')[0] != current_class:
                current_class = concept.split('_')[0]
                image_dir = os.path.join(self.working_dir, 'concepts', current_class, 'images')
                discovery_images = load_images_from_files(
                    [os.path.join(image_dir, file) for file in os.listdir(image_dir)],
                    do_shuffle=False)

        fig.update_layout(autosize=True,
                          width=(n_cols - 1) * 100,
                          height=n_rows * 100,
                          margin=dict(l=20, r=20, b=20, t=20)
                          )
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

        return fig
