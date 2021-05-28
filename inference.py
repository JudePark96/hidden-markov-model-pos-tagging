__author__ = 'Eunhwan Jude Park'
__email__ = 'judepark@{kookmin.ac.kr, jbnu.ac.kr}, jude.park.96@navercorp.com'

from dataset.data_utils import load_file
from hmm_model import viterbi_forward, viterbi_backward
from vocab.vocab_util import VocabUtil


if __name__ == '__main__':
    transition_prob = load_file('./rsc/probs/transition_prob.pkl')
    emission_prob = load_file('./rsc/probs/emission_prob.pkl')
    initial_prob = load_file('./rsc/probs/initial_prob.pkl')

    vocab = load_file('./rsc/vocab/vocab.pkl')
    states = vocab.tags

    states_dict = {key: idx for idx, key in enumerate(states)}

    while True:
        text = input('Input:')
        obs = text.split(' ')
        store, l = viterbi_forward(states, transition_prob, emission_prob, initial_prob, obs)
        best_seq, _ = viterbi_backward(states, store, l)
        print('original text', text)
        print('observation', obs)
        print('predicted tag', best_seq)
