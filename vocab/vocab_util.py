__author__ = 'Eunhwan Jude Park'
__email__ = 'judepark@{kookmin.ac.kr, jbnu.ac.kr}, jude.park.96@navercorp.com'

import pickle
from collections import defaultdict
from typing import List

from tqdm import tqdm

from dataset.data_utils import read_conllu, save_file, load_file


class VocabUtil(object):
    def __init__(self,
                 words: List[str],
                 tags: List[str]) -> None:
        super(VocabUtil, self).__init__()
        self.words = words
        self.pos_tags = tags

        self.vocab = []
        self.tags = []

    def build_vocab(self):
        words, tags = set(self.words), set(self.pos_tags)

        for word in words:
            self.vocab.append(word)

        for tag in tags:
            self.tags.append(tag)

    def convert_words_to_ids(self, sequence: List[str]) -> List[int]:
        if not isinstance(sequence, list):
            raise ValueError('Sequence has to be List[str].')

        return [0 if token not in self.vocab else self.vocab.index(token) for token in sequence]

    def convert_ids_to_words(self, sequence: List[int]) -> List[str]:
        if not isinstance(sequence, list):
            raise ValueError('Sequence has to be List[str].')

        return [0 if token_id > len(self.vocab) else self.vocab[token_id] for token_id in sequence]

    def convert_tags_to_ids(self, sequence: List[str]) -> List[int]:
        if not isinstance(sequence, list):
            raise ValueError('Sequence has to be List[str].')

        return [0 if tag not in self.tags else self.tags.index(tag) for tag in sequence]

    def convert_ids_to_tags(self, sequence: List[int]) -> List[str]:
        if not isinstance(sequence, list):
            raise ValueError('Sequence has to be List[int].')

        return [0 if tag_id > len(self.tags) else self.tags[tag_id] for tag_id in sequence]


if __name__ == '__main__':
    train = load_file('../rsc/preprocessed_dataset/train.pkl')
    dev = load_file('../rsc/preprocessed_dataset/dev.pkl')

    total = train + dev

    words, tags = [], []
    for data in total:
        for seq in data:
            words.append(seq[0])
            tags.append(seq[1])

    vocab = VocabUtil(words, tags)
    vocab.build_vocab()
    save_file('../rsc/vocab/vocab.pkl', vocab)

    vocab = load_file('../rsc/vocab/vocab.pkl')
    print(vocab.convert_words_to_ids('I have a dog'.split(' ')))