import argparse
import json
import os

from tqdm import tqdm

from dataset import clean_nl
from my_ast import MyAst, PathExtract
from utils.vocab import create_vocab

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', default='/home/tangze/data_set/', type=str)
parser.add_argument('-max_ast_len', default=250, type=int)
parser.add_argument('-process', action='store_true')
parser.add_argument('-make_vocab', action='store_true')


def skip_code_and_nl_with_skip_id(data_dir, output_dir, is_skipped):
    # skip data.
    nls = []
    with open(data_dir + 'nl.original', 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            if not is_skipped[line_index]:
                nls.append(line)

    codes = []
    with open(data_dir + 'code.seq', 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            if not is_skipped[line_index]:
                codes.append(line)

    # write to output_dir
    data_size = len(nls)

    with open(output_dir + 'nl.original', 'w') as f:
        for index, nl in tqdm(enumerate(nls), desc='skip nl'):
            nl = clean_nl(nl)
            nl = ' '.join(nl)
            if index < data_size-1:
                nl = nl + '\n'
            f.write(nl)

    with open(output_dir + 'code.seq', 'w') as f:
        for index, code in tqdm(enumerate(codes), desc='skip code'):
            f.write(code)


def process(data_dir, max_len, output_path):

    with open(data_dir + 'ast.original', 'r') as f:
        asts = []
        for line in f.readlines():
            ast_json = json.loads(line)
            asts.append(ast_json)

    is_skipped = PathExtract.collect_all_and_save(asts, output_path + 'paths.seq')

    asts = [ast for i, ast in enumerate(asts) if not is_skipped[i]]

    root_list = MyAst.process_ast(asts, split_leaf=False, max_size=max_len)

    MyAst.collect_matrices_and_save(root_list, output_path + 'un_split_matrices.npz', output_path + 'un_split_pot.seq')
    MyAst.collect_seq_and_save(root_list, output_path + 'un_split_sbt.seq', 'sbt')

    root_list = MyAst.process_ast(asts, split_leaf=True, max_size=max_len)

    MyAst.collect_matrices_and_save(root_list, output_path + 'split_matrices.npz', output_path + 'split_pot.seq')
    MyAst.collect_seq_and_save(root_list, output_path + 'split_sbt.seq', 'sbt')

    # skip code, nl with is_skipped
    skip_code_and_nl_with_skip_id(data_dir, output_path, is_skipped)


if __name__ == '__main__':
    args = parser.parse_args()
    data_set_dir = args.data_dir
    max_ast_len = args.max_ast_len

    languages = ['java/', 'py/']
    data_sets = ['test/', 'dev/', 'train/']

    if args.process:
        for lang in languages:
            for data_set in data_sets:
                data_path = data_set_dir + lang + data_set
                print('*' * 5, 'Process ', data_path, '*' * 5)
                processed_path = data_set_dir + 'processed/' + lang + data_set
                os.makedirs(processed_path, exist_ok=True)
                process(data_path, max_ast_len, processed_path)

    if args.make_vocab:
        for lang in languages:
            create_vocab(data_dir=data_set_dir + 'processed/' + lang)

