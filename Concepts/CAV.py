import os
import pickle as p
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional


class CAV:
    """ Discovery of Concept Activation Vectors that point in the direction of a concept in the activation space of a
    model. Based on Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F.,; Sayres, R. (2018).
    Interpretability beyond feature attribution: Quantitative Testing with Concept Activation Vectors (TCAV).
    35th International Conference on Machine Learning, ICML 2018, 6, 4186–4195.

    Trains CAVs based on images containing a concept and images not containing the concept. Saves this CAV in the form
    of a dictionary.
    """
    @staticmethod
    def load_cav(cav_path: str) -> 'CAV':
        """Loads an already created CAV in the form of a dictionary and returns a CAV instance.

        @param cav_path: Path to where the CAV is stored.
        @return: A CAV instance.

        """
        with open(cav_path, 'rb') as file:
            cav_dct = p.load(file)
        cav = CAV(cav_dct['bottleneck'], cav_dct['concept'], cav_dct['random_counterpart'])
        cav.cav = cav_dct['cav']
        cav.intercept = cav_dct['intercept']
        cav.norm = cav_dct['norm']
        cav.accuracy = cav_dct['accuracy']
        return cav

    def __init__(self, bottleneck, concept, random_counterpart):
        self.bottleneck = bottleneck
        self.concept = concept
        self.random_counterpart = random_counterpart
        self.cav = None
        self.intercept = None
        self.norm = None
        self.accuracy = None

    def save_cav(self, cav_dir: str) -> None:
        """Saves a CAV in the cav_dir in the form of a dictionary

        @param cav_dir: Path to the directory where the CAV will be stored.
        """
        cav_dct = {'bottleneck': self.bottleneck, 'concept': self.concept,
                   'random_counterpart': self.random_counterpart, 'cav': self.cav, 'intercept': self.intercept,
                   'norm': self.norm, 'accuracy': self.accuracy}

        file_path = os.path.join(cav_dir, f'{self.bottleneck}-{self.concept}-{self.random_counterpart}.pkl')
        with open(file_path, 'wb') as file:
            p.dump(cav_dct, file, protocol=-1)

    def train_cav(self, act_dct: Dict, param_dct: Dict):
        """Trains a CAV by fitting an SVM to predict the concept from the concept and its random counterpart.

        @param act_dct: Dictionary containing the activations of the concept and random counterpart. Of the form:
            {concept_name: activations_concept_imgs, random_counterpart_name:activations_random_imgs}.
        @param param_dct: Dictionary containing the parameters for training the SVM. Currently, uses default params.
        """
        # prepare data
        concepts_act, rnd_acts = act_dct[self.concept], act_dct[self.random_counterpart]
        min_imgs = min(len(concepts_act), len(rnd_acts))
        concepts_act = concepts_act[:min_imgs, :]
        rnd_acts = rnd_acts[:min_imgs, :]
        acts = np.concatenate((concepts_act, rnd_acts), axis=0)
        labels = [1] * min_imgs + [0] * min_imgs
        X_train, X_test, y_train, y_test = train_test_split(acts, labels, test_size=0.25, stratify=labels)

        # compute CAV
        svm = LinearSVC()
        svm.fit(X_train, y_train)
        self.accuracy = svm.score(X_test, y_test)
        self.cav = svm.coef_[0].reshape(1, -1)  # TODO maybe -1, 1 is nicer
        self.intercept = svm.intercept_[0]
        self.norm = np.linalg.norm(self.cav, ord=2)


def get_or_train_cav(concepts: List, bottleneck: str, act_dct: Dict, cav_dir: str,
                     param_dct: Optional[Dict] = None, ow: bool = False) -> 'CAV':
    """If exists loads a trained CAV, otherwise creates one.

    @param concepts: List [concept_name, random_counterpart_name].
    @param bottleneck: Name of the bottleneck layer.
    @param act_dct: Dictionary containing the activations of the examples of the concept and random counterpart.
        {concept_name: activations_concept_imgs, random_counterpart_name:activations_random_imgs}.
    @param cav_dir: Name of the directory where the CAV is or will be stored.
    @param param_dct: Parameters of the SVM that differentiates between concepts and the random counterpart.
    @param ow: If True, overwrite existing CAV.
    """
    cav_path = os.path.join(cav_dir, f'{bottleneck}-{concepts[0]}-{concepts[1]}.pkl')
    if os.path.exists(cav_path) and not ow:
        return CAV.load_cav(cav_path)
    else:
        cav = CAV(bottleneck, concepts[0], concepts[1])
        cav.train_cav(act_dct, param_dct)
        cav.save_cav(cav_dir)
        return cav
