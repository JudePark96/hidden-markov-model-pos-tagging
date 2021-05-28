__author__ = 'Eunhwan Jude Park'
__email__ = 'judepark@{kookmin.ac.kr, jbnu.ac.kr}, jude.park.96@navercorp.com'

import numpy as np


def viterbi_forward(states, transition_probs, emission_probs, initial_probs, observation):
    n = len(states)
    viterbi_list = []
    store = {}

    for state in states:
        viterbi_list.append(initial_probs[state] * emission_probs[f'{observation[0]}|{state}'])

    for i, ob in enumerate(observation):
        if i == 0:
            continue

        temp_list = [None] * n

        for j, state in enumerate(states):
            x = -1
            for k, prob in enumerate(viterbi_list):
                val = prob * transition_probs[f'{state}|{states[k]}'] * emission_probs[f'{ob}|{state}']
                if x < val:
                    x = val
                    store[f'{str(i)}-{state}'] = [states[k], val]
            temp_list[j] = x
        viterbi_list = [x for x in temp_list]

    return store, viterbi_list


def viterbi_backward(states, store, viterbi_list):
    num_states = len(states)
    n = len(store) // num_states
    best_seq = []
    best_seq_last = []
    x = states[np.argmax(np.asarray(viterbi_list))]
    best_seq.append(x)
    
    for i in range(n, 0, -1):
        val = store[str(i) + '-' + x][1]
        x = store[str(i) + '-' + x][0]
        best_seq = [x] + best_seq
        best_seq_last = [val] + best_seq_last

    return best_seq, best_seq_last
