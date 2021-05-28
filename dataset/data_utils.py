__author__ = 'Eunhwan Jude Park'
__email__ = 'judepark@{kookmin.ac.kr, jbnu.ac.kr}, jude.park.96@navercorp.com'

import pickle
from typing import List, Any
from conllu import TokenList, parse_incr
from tqdm import tqdm


def read_conllu(path: str) -> List[TokenList]:
    with open(path, 'r', encoding='utf-8') as f:
        parsed_data = [token_list for token_list in tqdm(parse_incr(f))]
        f.close()

    return parsed_data


def save_file(path: str, obj: Any) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        f.close()


def load_file(path: str) -> Any:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        f.close()

    return obj