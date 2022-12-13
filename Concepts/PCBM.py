import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from .ACE import ACE


# TODO make code into class
def PCBM(ace: 'ACE', imgs: np.ndarray, bottleneck: str, labels: np.ndarray):
    concept_space = ace.project_onto_concept_space(bottleneck, imgs)
    X_train, X_test, y_train, y_test = train_test_split(concept_space, labels, stratify=labels, test_size=250)
    lr = LogisticRegressionCV(penalty='elasticnet', solver='saga',
                              l1_ratios=[0,0.1,0.2,0.3,0.4,0.6,0.8,0.1], max_iter=10000)
    lr.fit(X_train, y_train)
    print(lr.score(X_test, y_test))
    return lr.coef_

