#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import ast
import dataclasses
import itertools
import os
import re
import string
from collections import defaultdict
from collections import namedtuple
from functools import lru_cache
from glob import glob
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import networkx as nx
from matplotlib import pyplot as plt


# ========================================================================= #
# Ast                                                                       #
# ========================================================================= #
from disent.util import TempNumpySeed


def ast_parse_tree(file):
    # load file
    with open(file, 'r') as f:
        data = f.read()
    # parse
    root = ast.parse(data)
    # add parents
    root.parent = None
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    # done
    return root, data.splitlines()


def ast_iter_parents(node):
    while hasattr(node, 'parent') and (node.parent is not None):
        node = node.parent
        yield node


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


Import = namedtuple('Import', 'module name')


@lru_cache()
def _file_to_import_str(file, root_dir=None):
    assert file.endswith('.py')
    # normalise paths
    if root_dir is not None:
        file = os.path.abspath(file)
        root_dir = os.path.abspath(root_dir) + '/'
        assert file.startswith(root_dir)
        file = file[len(root_dir):]
    assert not os.path.isabs(file), f'module path is not relative: {repr(file)} : {root_dir}'
    # get module
    module = file.replace('/', '.')[:-len('.py')]
    # done!
    return module


def file_to_import_str(file, root_dir=None, strip_mode: str = 'file'):
    return import_str_strip(_file_to_import_str(file, root_dir=root_dir), strip_mode=strip_mode)


@lru_cache()
def import_str_strip(import_path: str, strip_mode: str = 'file'):
    # strip end of module
    if strip_mode == 'no_init':
        if import_path.endswith('.__init__'):
            import_path = import_path[:-len('.__init__')]
    elif strip_mode == 'module':
        try:
            import_path = import_path[:import_path.rindex('.')]
        except:
            print(import_path)
            print(import_path)
            print(import_path)
            raise Exception
    elif strip_mode == 'file':
        pass
    else:
        raise KeyError(f'invalid remove mode: {repr(strip_mode)}')
    return import_path


class Module(object):

    def __init__(self, file: str, root_dir=None):
        self._file: str = file
        self._root_dir: str = os.getcwd() if (root_dir is None) else root_dir
        self._import_str: str = file_to_import_str(file, root_dir=self.root_dir)
        # load imports
        imports, scoped_imports = self._parse_imports(file)
        self._imports: List[Import] = imports
        self._scoped_imports: List[Import] = scoped_imports

    @property
    def file(self) -> str: return self._file
    @property
    def root_dir(self) -> str: return self._root_dir

    def get_import_str(self, strip_mode='file'):
        return import_str_strip(self._import_str, strip_mode=strip_mode)

    @property
    def import_no_init(self) -> str: return self.get_import_str(strip_mode='no_init')
    @property
    def import_file(self) -> str: return self.get_import_str(strip_mode='file')
    @property
    def import_module(self) -> str: return self.get_import_str(strip_mode='module')

    @property
    def imports(self) -> List[Import]: return list(self._imports)
    @property
    def scoped_imports(self) -> List[Import]: return list(self._scoped_imports)

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.file)})'

    def _parse_imports(self, file) -> Tuple[List[Import], List[Import]]:
        module, imports, scoped = self, [], []
        # parse
        root, data = ast_parse_tree(file)
        # visitor
        class Visitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for name in node.names:
                    self._add(node, Import(module=name.name, name=None))
            def visit_ImportFrom(self, node):
                for name in node.names:
                    self._add(node, module._make_from_import(file=file, data=data, node=node, module=node.module, name=name.name))
            def _add(self, node, imp):
                (imports if Module._ast_in_module_scope(node) else scoped).append(imp)
        # visit
        Visitor().visit(root)
        # done
        return imports, scoped

    def _make_from_import(self, file, data, node, module, name):
        # get line
        line = re.sub(' +', ' ', data[node.lineno - 1].strip())
        assert line.startswith('from ')
        line = line[len('from '):]
        assert line[0] not in string.whitespace
        # check if relative
        if line[0] == '.':
            module = file_to_import_str(file, root_dir=self.root_dir, strip_mode='module') + f'.{module}'
        return Import(module=module, name=name)

    @staticmethod
    def _ast_in_module_scope(node):
        for parent in ast_iter_parents(node):
            if isinstance(parent, (ast.AsyncFunctionDef, ast.FunctionDef)):
                return False
        return True


class Modules(object):

    def __init__(self, modules):
        self._modules: List[Module] = list(modules)

    def __repr__(self):
        return f'Modules<num={self._modules}>'

    @property
    def modules(self):
        return list(self._modules)

    @staticmethod
    def from_files(file_paths: Union[str, Sequence[str]], root_dir=None) -> 'Modules':
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        # filter paths, but keep originals
        files_sets = defaultdict(set)
        for file in file_paths:
            files_sets[os.path.abspath(file)].add(file)
        assert all(len(files) == 1 for files in files_sets.values())
        # make _modules
        return Modules([Module(files.pop(), root_dir=root_dir) for files in files_sets.values()])

    @staticmethod
    def from_glob(glob_paths: Union[str, Sequence[str]] = '**/*.py', recursive=True, root_dir=None) -> 'Modules':
        if isinstance(glob_paths, str):
            glob_paths = [glob_paths]
        # glob all paths
        module_files = []
        for glob_path in glob_paths:
            module_files.extend(glob(glob_path, recursive=recursive))
        # make _modules
        return Modules.from_files(module_files, root_dir=root_dir)

    @staticmethod
    def from_dirs(dir_paths: Union[str, Sequence[str]], recursive=True, root_dir=None) -> 'Modules':
        if isinstance(dir_paths, str):
            dir_paths = [dir_paths]
        # normalise dirs
        dir_paths = [dir_path.rstrip('/') for dir_path in dir_paths]
        glob_paths = [(f'{dir_path}/**/*.py' if recursive else f'{dir_path}/*.py') for dir_path in dir_paths]
        # return paths
        return Modules.from_glob(glob_paths, recursive=recursive, root_dir=root_dir)

    def make_graph(self, include_scoped=False, module_type='file', startswith=None):
        if startswith is None:
            startswith = ''
        # done
        G = nx.DiGraph()
        for module in self._modules:
            # skip module
            if not module.get_import_str().startswith(startswith):
                continue
            # get imports
            imports = (module.imports + module.scoped_imports) if include_scoped else module.imports
            # add edges
            for imp in imports:
                # skip module
                if not imp.module.startswith(startswith):
                    continue
                # add edge
                G.add_edge(import_str_strip(imp.module, ), module.get_import_str(strip_mode=module_type))
        return G

    @staticmethod
    def graph_find_cycles(graph: nx.DiGraph):
        return list(nx.simple_cycles(graph))

    @staticmethod
    def graph_make_plot(graph: nx.DiGraph):
        fig, ax = plt.subplots(figsize=(15, 15))
        nx.draw_kamada_kawai(graph, with_labels=True, font_weight='bold', ax=ax)
        return fig, ax

    @staticmethod
    def graph_save_visualisation(graph: nx.DiGraph, save_file: str):
        # check
        assert save_file.endswith('.html')
        os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
        # visualise
        from pyvis.network import Network
        net = Network(notebook=False, directed=True, width='100%', height='100%')
        net.from_nx(graph)
        net.show(save_file)

# ========================================================================= #
# Graph Util                                                                #
# ========================================================================= #

# def get_column_widths(table):
#     lengths = defaultdict(int)
#     for row in table:
#         for i, entry in enumerate(row):
#             lengths[i] = max(lengths[i], len(entry))
#     return list(lengths[i] for i in sorted(lengths.keys()))
#
# def pad_table_columns(table):
#     lengths = get_column_widths(table)
#     return [[f'{entry:{l}s}' for l, entry in zip(lengths, row)] for row in table]
#
# for row in pad_table_columns(imports):
#     print(' | '.join(row))


def filtered_graph_edges(graph: nx.DiGraph, starts_with, return_removed=False):
    graph, removed = graph.copy(), []
    for module, imp in list(graph.edges):
        if not module.startswith(starts_with) or not imp.startswith(starts_with):
            graph.remove_edge(module, imp)
            removed.append((module, imp))
    if return_removed:
        return graph, removed
    return graph


# ========================================================================= #
# Loader                                                                    #
# ========================================================================= #

@lru_cache()
def normalise_import_str(import_str: str, root_dir: str):
    file = import_str.replace('.', '/')
    file = os.path.abspath(os.path.join(root_dir, file))
    # check if module or file
    is_module = os.path.isdir(file)
    is_file = os.path.isfile(file + '.py')
    # check which kind it is
    if is_module and is_file:
        raise RuntimeError
    elif not is_module and not is_file:
        raise RuntimeError
    elif is_module:
        import_str += '.__init__'
        file += '/__init__.py'
        if not os.path.isfile(file):
            raise FileNotFoundError(f'module does not have an __init__.py file: {repr(import_str)}')
    elif is_file:
        pass
    else:
        raise Exception
    # done
    return import_str



if __name__ == '__main__':
    with TempNumpySeed(777):
        ROOT = os.path.abspath(os.path.join(__file__, '../..'))
        DISENT = os.path.abspath(os.path.join(ROOT, 'disent'))
        modules = Modules.from_dirs(DISENT, root_dir=ROOT)

        # visualise
        for import_prefix, module_type in itertools.product(['disent', 'disent.frameworks'], ['file']):
            # make string
            suffix = '+'.join(s for s in [module_type, import_prefix if import_prefix else 'ANY'])
            save_file = os.path.join(ROOT, f'out/imports/imports_{suffix}.html')
            # make graphs
            graph = modules.make_graph(include_scoped=False, module_type=module_type, startswith=import_prefix)
            # filter and replace edges
            for imp, mod in list(graph.edges):
                n_imp, n_mod = normalise_import_str(imp, ROOT), normalise_import_str(mod, ROOT)
                graph.remove_edge(imp, mod)
                graph.add_edge(n_imp, n_mod)
            # print cycles
            print(suffix)
            for cycle in sorted(Modules.graph_find_cycles(graph)):
                print('--', ' -> '.join(cycle))
            print()
            # make new graph
            G = nx.DiGraph()
            for imp, mod in graph.edges:
                imp = import_str_strip(imp, strip_mode='module')
                mod = import_str_strip(mod, strip_mode='module')
                if imp != mod:
                    G.add_edge(imp, mod)
            # visualise
            Modules.graph_save_visualisation(G, save_file=save_file)

