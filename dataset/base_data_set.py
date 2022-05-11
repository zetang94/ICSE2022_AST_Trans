import random

import torch
import numpy as np
import torch.utils.data as data
import re
import wordninja
import string
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.data.dataloader import Collater
from tqdm import tqdm
from utils import PAD, UNK
from torch.utils.data.dataset import T_co
punc = string.punctuation

__all__ = ['BaseASTDataSet', 'clean_nl', 'subsequent_mask', 'make_std_mask']


class BaseASTDataSet(data.Dataset):
    def __init__(self, config, data_set_name):
        super(BaseASTDataSet, self).__init__()
        self.data_set_name = data_set_name
        print('loading ' + data_set_name + ' data...')
        data_dir = config.data_dir + '/' + data_set_name + '/'

        self.ignore_more_than_k = config.is_ignore
        self.max_rel_pos = config.max_rel_pos
        self.max_src_len = config.max_src_len
        self.max_tgt_len = config.max_tgt_len

        ast_path = data_dir + 'split_pot.seq' if config.is_split else 'un_split_pot.seq'
        matrices_path = data_dir + 'split_matrices.npz' if config.is_split else 'un_split_matrices.npz'

        self.ast_data = load_list(ast_path)
        self.nl_data = load_seq(data_dir + 'nl.original')
        self.matrices_data = load_matrices(matrices_path)

        self.data_set_len = len(self.ast_data)
        self.src_vocab = config.src_vocab
        self.tgt_vocab = config.tgt_vocab
        self.collector = Collater([], [])

    def collect_fn(self, batch):
        return self.collector.collate(batch)

    def __len__(self):
        return self.data_set_len

    def __getitem__(self, index) -> T_co:
        pass

    def convert_ast_to_tensor(self, ast_seq):
        ast_seq = ast_seq[:self.max_src_len]
        return word2tensor(ast_seq, self.max_src_len, self.src_vocab)

    def convert_nl_to_tensor(self, nl):
        nl = nl[:self.max_tgt_len - 2]
        nl = ['<s>'] + nl + ['</s>']
        return word2tensor(nl, self.max_tgt_len, self.tgt_vocab)


def word2tensor(seq, max_seq_len, vocab):
    seq_vec = [vocab.w2i[x] if x in vocab.w2i else UNK for x in seq]
    seq_vec = seq_vec + [PAD for i in range(max_seq_len - len(seq_vec))]
    seq_vec = torch.tensor(seq_vec, dtype=torch.long)
    return seq_vec


def load_list(file_path):
    _data = []
    print(f'loading {file_path}...')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            _data.append(eval(line))
    return _data


def load_seq(file_path):
    data_ = []
    print(f'loading {file_path} ...')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            data_.append(line.split())
    return data_


def load_matrices(file_path):
    print('loading matrices...')
    matrices = np.load(file_path, allow_pickle=True)
    return matrices


def clean_nl(s):
    s = s.strip()
    if s[-1] == ".":
        s = s[:-1]
    s = s.split(". ")[0]
    s = re.sub("[<].+?[>]", "", s)
    s = re.sub("[\[\]\%]", "", s)
    s = s[0:1].lower() + s[1:]
    processed_words = []
    for w in s.split():
        if w not in punc:
            processed_words.extend(wordninja.split(w))
        else:
            processed_words.append(w)
    return processed_words


def subsequent_mask(size):
    attn_shape = (1, size, size)
    sub_sequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(sub_sequent_mask) != 0


def make_std_mask(nl, pad):
    "Create a mask to hide padding and future words."
    nl_mask = (nl == pad).unsqueeze(-2)
    nl_mask = nl_mask | Variable(
        subsequent_mask(nl.size(-1)).type_as(nl_mask.data))
    return nl_mask



