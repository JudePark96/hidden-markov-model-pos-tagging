__author__ = 'Eunhwan Jude Park'
__email__ = 'judepark@{kookmin.ac.kr, jbnu.ac.kr}, jude.park.96@navercorp.com'

from dataset.data_utils import load_file
from hmm_model import viterbi_forward, viterbi_backward
from vocab.vocab_util import VocabUtil

import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model():
    transition_prob = load_file('./rsc/probs/transition_prob.pkl')
    emission_prob = load_file('./rsc/probs/emission_prob.pkl')
    initial_prob = load_file('./rsc/probs/initial_prob.pkl')

    vocab = load_file('./rsc/vocab/vocab.pkl')
    states = vocab.tags

    return transition_prob, emission_prob, initial_prob, vocab, states


if __name__ == '__main__':
    transition_prob, emission_prob, initial_prob, vocab, states = load_model()
    st.title('Hidden Markov Model w/ POS Tagging')
    text = st.text_area('Sentence Input:')

    st.markdown('## Sentence')
    st.write(text)

    if text:
        text = text.replace('\n', '')
        st.markdown("## POS Tagging Result")
        with st.spinner('processing..'):
            obs = text.split(' ')
            store, l = viterbi_forward(states, transition_prob, emission_prob, initial_prob, obs)
            best_seq, _ = viterbi_backward(states, store, l)

        st.write(' '.join(best_seq))
