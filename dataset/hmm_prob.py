__author__ = 'Eunhwan Jude Park'
__email__ = 'judepark@{kookmin.ac.kr, jbnu.ac.kr}, jude.park.96@navercorp.com'

import numpy as np
from tqdm import tqdm

from dataset.data_utils import load_file, save_file
from vocab.vocab_util import VocabUtil


class HMMProbUtil(object):
    def __init__(self,
                 vocab: VocabUtil) -> None:
        super(HMMProbUtil, self).__init__()
        self.vocab = vocab

    def get_transition_prob(self, data) -> dict:
        states = self.vocab.tags
        prob = np.zeros((len(states), len(states)))

        for sentence in data:
            for i in range(len(sentence)):
                if i == 0:
                    continue
                prob[states.index(sentence[i - 1][1])][states.index(sentence[i][1])] += 1
        prob /= np.sum(prob, keepdims=True, axis=1)

        states = self.vocab.tags
        state_dict = {}
        for i, state in enumerate(states):
            state_dict[i] = state

        return_prob = {}

        for i in range(prob.shape[0]):
            for j in range(prob.shape[1]):
                return_prob[state_dict[j]+'|'+state_dict[i]] = prob[i][j]

        return return_prob

    def get_emission_prob(self, data) -> dict:
        vocab_len, states_len = len(self.vocab.vocab), len(self.vocab.tags)
        prob = np.zeros((vocab_len, states_len))

        for sentence in tqdm(data):
            for word, tag in sentence:
                prob[self.vocab.vocab.index(word)][self.vocab.tags.index(tag)] += 1

        sum_prob = np.sum(prob, keepdims=True, axis=0)
        prob = np.divide(prob, sum_prob, out=np.zeros_like(prob), where=sum_prob != 0)

        states = self.vocab.tags
        state_dict = {}
        for i, state in enumerate(states):
            state_dict[i] = state

        return_prob = {}

        for i in range(prob.shape[0]):
            for j in range(prob.shape[1]):
                if prob[i][j] == 0.0:
                    return_prob[self.vocab.vocab[i]+'|'+state_dict[j]] = 5e-5
                else:
                    return_prob[self.vocab.vocab[i]+'|'+state_dict[j]] = prob[i][j]

        return return_prob

    def get_initial_prob(self, data) -> dict:
        initial_list = [0] * len(self.vocab.tags)
        for sentence in tqdm(data):
            tag = sentence[0][1]
            initial_list[self.vocab.tags.index(tag)] += 1
        initial_list = np.asarray(initial_list)
        initial_list = initial_list / np.sum(initial_list, keepdims=True)
        initial_probs = {}
        for i, state in enumerate(self.vocab.tags):
            initial_probs[state] = initial_list[i]
        return initial_probs


if __name__ == '__main__':
    train = load_file('../rsc/preprocessed_dataset/train.pkl')
    dev = load_file('../rsc/preprocessed_dataset/dev.pkl')
    vocab = load_file('../rsc/vocab/vocab.pkl')
    hmm = HMMProbUtil(vocab)

    data = train #+ dev

    trans_prob = hmm.get_transition_prob(data)
    emis_prob = hmm.get_emission_prob(data)
    train_init_prob = hmm.get_initial_prob(data)
    dev_init_prob = hmm.get_initial_prob(dev)

    save_file('../rsc/probs/transition_prob.pkl', trans_prob)
    save_file('../rsc/probs/emission_prob.pkl', emis_prob)
    save_file('../rsc/probs/train_initial_prob.pkl', train_init_prob)
    save_file('../rsc/probs/dev_initial_prob.pkl', dev_init_prob)

    print('done')
