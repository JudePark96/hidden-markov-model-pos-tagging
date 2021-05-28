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


if __name__ == '__main__':
    main()
