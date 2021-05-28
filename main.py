__author__ = 'Eunhwan Jude Park'
__email__ = 'judepark@{kookmin.ac.kr, jbnu.ac.kr}, jude.park.96@navercorp.com'

import logging
from typing import List, Dict

import numpy as np
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from dataset.data_utils import load_file
from hmm_model import viterbi_forward, viterbi_backward
from vocab.vocab_util import VocabUtil

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_observation(data):
    observations, labels = [], []

    for sentence in data:
        observations.append([word for word, tag in sentence])
        labels.append([tag for word, tag in sentence])

    return observations, labels


def evaluate(predict_tags: List[str],
             label_tags: List[str],
             tag_dict: Dict[str, int]) -> float:
    convert_predict = ([tag_dict[tag] for tag in predict_tags])
    convert_label = ([tag_dict[tag] for tag in label_tags])
    f1 = f1_score(np.array(convert_label), np.array(convert_predict), average='weighted')
    return f1


def main():
    train = load_file('./rsc/preprocessed_dataset/train.pkl')
    dev = load_file('./rsc/preprocessed_dataset/dev.pkl')

    transition_prob = load_file('./rsc/probs/transition_prob.pkl')
    emission_prob = load_file('./rsc/probs/emission_prob.pkl')
    train_initial_prob = load_file('./rsc/probs/train_initial_prob.pkl')
    dev_initial_prob = load_file('./rsc/probs/train_initial_prob.pkl')

    vocab = load_file('./rsc/vocab/vocab.pkl')
    states = vocab.tags

    states_dict = {key: idx for idx, key in enumerate(states)}
    train_obs, train_label = get_observation(train)
    dev_obs, dev_label = get_observation(dev)

    train_f1_pred, train_f1_true = [], []
    dev_f1_pred, dev_f1_true = [], []

    for obs, label in tqdm(zip(train_obs, train_label), total=len(train_obs)):
        store, l = viterbi_forward(states, transition_prob, emission_prob, train_initial_prob, obs)
        best_seq, _ = viterbi_backward(states, store, l)
        train_f1_pred.extend(best_seq)
        train_f1_true.extend(label)

    logger.info('train f1 report')
    logger.info(classification_report(train_f1_true, train_f1_pred, states))

    for obs, label in tqdm(zip(dev_obs, dev_label), total=len(dev_obs)):
        store, l = viterbi_forward(states, transition_prob, emission_prob, dev_initial_prob, obs)
        best_seq, _ = viterbi_backward(states, store, l)
        dev_f1_pred.extend(best_seq)
        dev_f1_true.extend(label)

    logger.info('dev f1 report')
    logger.info(classification_report(dev_f1_true, dev_f1_pred, states))

    """
                precision    recall    f1       support
    Train
        PART       0.86      0.93      0.89      5567
           _       0.98      0.83      0.90      2616
         SYM       0.71      0.81      0.76       698
         DET       0.86      0.97      0.91     16309
         AUX       0.85      0.96      0.90     12435
       CCONJ       0.99      0.98      0.98      6703
         ADJ       0.86      0.81      0.84     13127
         ADP       0.88      0.93      0.90     17729
       SCONJ       0.77      0.74      0.76      4502
        VERB       0.87      0.82      0.84     23001
       PROPN       0.87      0.74      0.80     12316
        INTJ       0.92      0.89      0.90       688
        NOUN       0.89      0.83      0.86     34797
         NUM       0.94      0.88      0.91      4112
       PUNCT       0.93      0.99      0.96     23620
         ADV       0.84      0.78      0.81      9709
        PRON       0.90      0.96      0.93     18586
           X       0.71      0.82      0.76       709

    accuracy                           0.88    207224
   macro avg       0.87      0.87      0.87    207224
weighted avg       0.88      0.88      0.88    207224

    Dev
        PART       0.85      0.92      0.88       633
           _       1.00      0.73      0.85       360
         SYM       0.61      0.66      0.63        80
         DET       0.78      0.97      0.86      1899
         AUX       0.80      0.95      0.87      1514
       CCONJ       0.99      0.98      0.98       780
         ADJ       0.84      0.76      0.80      1874
         ADP       0.86      0.92      0.89      2030
       SCONJ       0.71      0.70      0.70       466
        VERB       0.82      0.77      0.80      2762
       PROPN       0.80      0.43      0.56      1787
        INTJ       0.84      0.60      0.70       115
        NOUN       0.81      0.82      0.81      4198
         NUM       0.85      0.63      0.73       383
       PUNCT       0.91      0.98      0.95      3077
         ADV       0.83      0.74      0.78      1187
        PRON       0.82      0.96      0.88      2220
           X       0.42      0.18      0.26       146

    accuracy                           0.83     25511
   macro avg       0.81      0.76      0.77     25511
weighted avg       0.83      0.83      0.83     25511
    """

if __name__ == '__main__':
    main()
