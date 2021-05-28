__author__ = 'Eunhwan Jude Park'
__email__ = 'judepark@{kookmin.ac.kr, jbnu.ac.kr}, jude.park.96@navercorp.com'

from typing import List, Tuple, Any
from tqdm import tqdm
from dataset.data_utils import read_conllu, save_file


def preprocess_conllu(path: str) -> List[List[Tuple[Any, Any]]]:
    dataset = read_conllu(path)
    output = []
    for data in tqdm(dataset):
        output.append([(j['form'], j['upos']) for j in data])

    return output


if __name__ == '__main__':
    train = preprocess_conllu(path='../rsc/dataset/en_ewt-ud-train.conllu')
    dev = preprocess_conllu(path='../rsc/dataset/en_ewt-ud-dev.conllu')

    save_file('../rsc/preprocessed_dataset/train.pkl', train)
    save_file('../rsc/preprocessed_dataset/dev.pkl', dev)

