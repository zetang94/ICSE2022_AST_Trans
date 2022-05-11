import logging
import os
import pickle
import unicodedata
from collections import Counter

from tqdm import tqdm

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

log = logging.getLogger()

__all__ = ['Vocab', 'PAD', 'BOS', 'EOS', 'UNK', 'EOS_WORD', 'BOS_WORD', 'PAD_WORD',
           'create_vocab', 'load_vocab']


class Vocab(object):
    def __init__(self, need_bos, file_path):
        if not need_bos:
            self.w2i = {PAD_WORD: PAD,
                        UNK_WORD: UNK}
            self.i2w = {PAD: PAD_WORD,
                        UNK: UNK_WORD}
        else:
            self.w2i = {PAD_WORD: PAD,
                        UNK_WORD: UNK,
                        BOS_WORD: BOS,
                        EOS_WORD: EOS}
            self.i2w = {PAD: PAD_WORD,
                        UNK: UNK_WORD,
                        BOS: BOS_WORD,
                        EOS: EOS_WORD}
        self.file_path = file_path

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def size(self):
        return len(self.w2i)

    def add(self, token):
        token = self.normalize(token)
        if token not in self.w2i:
            index = len(self.w2i)
            self.w2i[token] = index
            self.i2w[index] = token

    def add_tokens(self, tokens):
        for token in tokens:
            self.add(token)

    def generate_dict(self, tokens, max_vocab_size=-1):
        word_counter = Counter([x for c in tokens for x in c])
        if max_vocab_size < 0:
            words = [x[0] for x in word_counter.most_common()]
        else:
            words = [x[0] for x in word_counter.most_common(max_vocab_size - len(self.w2i))]
        self.add_tokens(words)

        self.save()

    def save(self):
        pickle.dump(self.w2i, open(self.file_path, "wb"))

    def load(self):
        self.w2i = pickle.load(open(self.file_path, "rb"))
        self.i2w = {v: k for k, v in self.w2i.items()}


def load_vocab(data_dir, is_split, data_type):
    log.info(f'load vocab from {data_dir}, is_split = {is_split}')
    split_str = 'split_ast_vocab.pkl' if is_split else 'un_split_ast_vocab.pkl'
    if data_type in ['sbt', 'path', 'pot', 'gnn']:
        src_vocab = Vocab(need_bos=False,
                          file_path=data_dir + '/vocab/' + split_str)
        src_vocab.load()
    elif data_type in ['code']:
        src_vocab = Vocab(need_bos=False, file_path=data_dir + '/vocab/' + 'code_vocab.pkl')
        src_vocab.load()
    nl_vocab = Vocab(need_bos=True, file_path=data_dir + '/vocab/' + 'nl_vocab.pkl')
    nl_vocab.load()
    return src_vocab, nl_vocab


def create_vocab(data_dir):
    # create vocab
    log.info('init vocab')
    output_dir = data_dir + 'vocab/'
    os.makedirs(output_dir, exist_ok=True)

    un_split_ast_tokens = []
    with open(data_dir + 'train/' + 'un_split_sbt.seq', 'r') as f:
        for line in tqdm(f.readlines(), desc='loading ast from train ....'):
            un_split_ast_tokens.append(eval(line))
    with open(data_dir + 'dev/' + 'un_split_sbt.seq', 'r') as f:
        for line in tqdm(f.readlines(), desc='loading ast from dev ....'):
            un_split_ast_tokens.append(eval(line))

    un_split_ast_vocab = Vocab(need_bos=False, file_path=output_dir + 'un_split_ast_vocab.pkl')
    un_split_ast_vocab.generate_dict(un_split_ast_tokens)

    split_ast_tokens = []
    with open(data_dir + 'train/' + 'split_sbt.seq', 'r') as f:
        for line in tqdm(f.readlines(), desc='loading ast from train ....'):
            split_ast_tokens.append(eval(line))
    with open(data_dir + 'dev/' + 'split_sbt.seq', 'r') as f:
        for line in tqdm(f.readlines(), desc='loading ast from dev ....'):
            split_ast_tokens.append(eval(line))

    split_ast_vocab = Vocab(need_bos=False, file_path=output_dir + 'split_ast_vocab.pkl')
    split_ast_vocab.generate_dict(split_ast_tokens)

    code_tokens = []
    with open(data_dir + 'train/' + 'code.seq', 'r') as f:
        for line in tqdm(f.readlines(), desc='loading code from train ....'):
            code_tokens.append(line.split())

    with open(data_dir + 'dev/' + 'code.seq', 'r') as f:
        for line in tqdm(f.readlines(), desc='loading code from dev ....'):
            code_tokens.append(line.split())

    code_vocab = Vocab(need_bos=False, file_path=output_dir + 'code_vocab.pkl')
    code_vocab.generate_dict(code_tokens)

    nl_tokens = []
    with open(data_dir + 'train/nl.original', 'r') as f:
        for line in tqdm(f.readlines(), desc='loading nl from train ....'):
            nl_tokens.append(line.split())
    with open(data_dir + 'dev/nl.original', 'r') as f:
        for line in tqdm(f.readlines(), desc='loading nl from dev ....'):
            nl_tokens.append(line.split())
    nl_vocab = Vocab(need_bos=True, file_path=output_dir + 'nl_vocab.pkl')
    nl_vocab.generate_dict(nl_tokens)
    log.info(f"un_split ast vocab size: {len(un_split_ast_vocab.w2i)} \n"
             f"split ast vocab size: {len(split_ast_vocab.w2i)} \n"
             f"code vocab size: {len(code_vocab.w2i)} \n"
             f"nl vocab size: {len(nl_vocab.w2i)}")


