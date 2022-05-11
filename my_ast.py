import copy
import itertools
import re
import joblib
import numpy as np
from tqdm import tqdm

__all__ = ['split_variable', 'PathExtract', 'MyAst']


n_jobs = 8


class Node:
    def __init__(self, label="", parent=None, is_simple_name=False, children=[]):
        self.label = label
        self.parent = parent
        self.children = children
        self.is_simple_name = is_simple_name


class MyAst:
    @staticmethod
    def process_ast(asts, split_leaf=False, max_size=-1):
        parallel = joblib.Parallel(n_jobs=n_jobs)
        func = joblib.delayed(MyAst.__process)

        root_nodes = parallel(func(ast, split_leaf, max_size)
                              for ast in tqdm(asts,
                                              desc=f'process AST: split leaf {split_leaf} size {max_size}'))
        return root_nodes

    @staticmethod
    def collect_seq_and_save(root_nodes, output_file, seq_type):
        parallel = joblib.Parallel(n_jobs=n_jobs)
        if seq_type == 'sbt':
            func = joblib.delayed(MyAst.__get_sbt_seq)
        elif seq_type == 'pot':
            func = joblib.delayed(MyAst.__get_pot_seq)
        else:
            raise Exception('Invalid seq_type, must be in [sbt, pot]')

        seqs = parallel(func(root_node) for root_node in
                        tqdm(root_nodes, desc='generate ' + seq_type))

        with open(output_file, 'w') as f:
            for line_index, line in enumerate(seqs):
                f.write(str(line) + ('' if line_index == len(seqs) - 1 else '\n'))

    @staticmethod
    def collect_matrices_and_save(root_nodes, output_matrices_file, output_pot_file):
        parallel = joblib.Parallel(n_jobs=n_jobs)
        func = joblib.delayed(MyAst.__get_matrices)

        matrices = parallel(func(root_node) for root_node in
                            tqdm(root_nodes, desc='generate matrices'))
        pot_seq, par_edges, bro_edges = list(zip(*matrices))

        np.savez(output_matrices_file,
                 parent=list(par_edges), brother=list(bro_edges))

        with open(output_pot_file, 'w') as f:
            for line_index, line in enumerate(pot_seq):
                f.write(str(line) + ('' if line_index == len(pot_seq) - 1 else '\n'))

    @staticmethod
    def __process(ast_json, split_leaf=False, max_size=-1):
        node_num = len(ast_json)
        node_list = [copy.deepcopy(Node()) for i in range(node_num)]

        for i in range(node_num):
            node_attr = ast_json[i]
            node = node_list[i]

            node.label = node_attr['type']
            if 'value' in node_attr:
                _value = node_attr['value']
                if is_num(_value):
                    node.children = [Node(label=_value, parent=node, is_simple_name=True, children=[])]
                else:
                    if split_leaf:
                        words = split_variable(_value)
                        node.children = [Node(label=w, parent=node, is_simple_name=True, children=[]) for w in words]
                    else:
                        node.children = [Node(label=_value, parent=node, is_simple_name=True, children=[])]

            if 'children' in node_attr:
                for child_id in node_attr['children']:
                    node_list[child_id].parent = node
                    if i == 0:
                        """This can prevent important nodes such as parameters or return values 
                        from being ignored when cutting the tree. """
                        node.children.insert(0, node_list[child_id])
                    else:
                        node.children.append(node_list[child_id])

        if max_size > 0:
            MyAst.__sub_tree(node_list[0], max_size)
        return node_list[0]

    @staticmethod
    def __sub_tree(root_node, max_size, i=0):
        root_node.num = i
        i = i + 1
        if i > max_size:
            return -1
        else:
            for j, child in enumerate(root_node.children):
                i = MyAst.__sub_tree(child, max_size, i)
                if i == -1:
                    root_node.children = root_node.children[:j]
                    return -2
                if i == -2:
                    root_node.children = root_node.children[:j + 1]
                    return i
            return i

    @staticmethod
    def __get_root_first_seq(root_node):
        li = [root_node]
        for child in root_node.children:
            li += MyAst.__get_root_first_seq(child)
        return li

    @staticmethod
    def __get_pot_seq(root_node):
        root_first_seq = MyAst.__get_root_first_seq(root_node)
        root_first_labels = [node.label for node in root_first_seq]
        return root_first_labels

    @staticmethod
    def __get_sbt_seq(root):
        sbt_seq = ["(", root.label]
        for child in root.children:
            sbt_seq += MyAst.__get_sbt_seq(child)
        sbt_seq += [")", root.label]
        return sbt_seq

    @staticmethod
    def __get_matrices(root_node):
        root_first_seq = MyAst.__get_root_first_seq(root_node)
        root_first_labels = [node.label for node in root_first_seq]

        distance_map = {}
        brother_map = {}

        parent_path_list = []
        brother_path_list = []

        for node in root_first_seq:
            if len(node.children) == 0:
                path = [node.num]
                n = node
                while n.parent is not None:
                    path.append(n.parent.num)
                    n = n.parent
                    parent_path_list.append(list(reversed(path)))
            else:
                brother_path_list.append([child.num for child in node.children])

        for path in parent_path_list:
            distance_map.update(MyAst.__get_distance_pairs(path))

        for path in brother_path_list:
            brother_map.update(MyAst.__get_distance_pairs(path))

        return root_first_labels, distance_map, brother_map

    @staticmethod
    def __get_distance_pairs(path):
        node_num = len(path)
        distance_pairs = {}
        if node_num >= 2:
            for i in range(node_num - 1):
                for j in range(i + 1, node_num):
                    distance_pairs[(path[i], path[j])] = j - i
        return distance_pairs


class PathExtract:
    @staticmethod
    def __terminals(ast, node_index=0):
        stack, paths = [], []

        def dfs(v):
            stack.append(v)
            v_node = ast[v]

            if 'value' in v_node:
                v_value = v_node['value']

                if is_num(v_value):
                    # keep the same as code2seq, replace number with NUM
                    paths.append((stack.copy(), 'NUM'))
                else:
                    paths.append((stack.copy(), v_value))

            if 'children' in v_node:
                for child in v_node['children']:
                    dfs(child)

            stack.pop()

        dfs(node_index)

        return paths

    @staticmethod
    def __merge_terminals2_paths(v_path, u_path):
        s, n, m = 0, len(v_path), len(u_path)
        while s < min(n, m) and v_path[s] == u_path[s]:
            s += 1

        prefix = list(reversed(v_path[s:]))
        lca = v_path[s - 1]
        suffix = u_path[s:]

        return prefix, lca, suffix

    @staticmethod
    def __raw_tree_paths(ast, node_index):
        tnodes = PathExtract.__terminals(ast, node_index)

        tree_paths = []
        for (v_path, v_value), (u_path, u_value) in itertools.combinations(
                iterable=tnodes,
                r=2,
        ):
            prefix, lca, suffix = PathExtract.__merge_terminals2_paths(v_path, u_path)
            if (len(prefix) + 1 + len(suffix) <= 8) \
                    and (abs(len(prefix) - len(suffix)) <= 2):
                path = prefix + [lca] + suffix
                tree_path = v_value, path, u_value
                tree_paths.append(tree_path)

        return tree_paths

    @staticmethod
    def __collect_sample(ast, fd_index=0):
        tree_paths = PathExtract.__raw_tree_paths(ast, fd_index)
        contexts = []
        for tree_path in tree_paths:
            start, connector, finish = tree_path
            start, finish = split_variable(start), split_variable(finish)

            start = '|'.join(start)
            finish = '|'.join(finish)
            connector = '|'.join(ast[v]['type'] for v in connector)

            context = f'{start},{connector},{finish}'
            if context == '':
                continue
            contexts.append(context)

        if len(contexts) == 0:
            return None

        context = ';'.join(contexts)
        if len(context.split(';')) != len(contexts):
            raise Exception('context should not be concat with ;')

        return f'{context}'

    @staticmethod
    def collect_all_and_save(asts, output_file):
        parallel = joblib.Parallel(n_jobs=n_jobs)
        func = joblib.delayed(PathExtract.__collect_sample)

        samples = parallel(func(ast) for ast in tqdm(asts, desc='generate path'))
        samples = list(samples)

        is_skipped = []

        with open(output_file, 'w') as f:
            for line_index, line in enumerate(samples):
                if line is None:
                    is_skipped.append(True)
                else:
                    is_skipped.append(False)
                    f.write(line + ('' if line_index == len(samples) - 1 else '\n'))

        return is_skipped


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def split_variable(name):
    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]

    blocks = []
    for underscore_block in name.split('_'):
        blocks.extend(camel_case_split(underscore_block))

    return [block.lower() for block in blocks]



