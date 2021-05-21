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
def _module_file_to_str(file, root_dir=None, remove_init=False):
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
    # assert all(str.isidentifier(token) for token in module.split('.'))
    # remove __init__
    if remove_init:
        if module.endswith('.__init__'):
            module = module[:-len('.__init__')]
    return module


class Module(object):

    def __init__(self, file: str, root_dir=None):
        self._file: str = file
        self._root_dir: str = os.getcwd() if (root_dir is None) else root_dir
        self._module: str = _module_file_to_str(file, root_dir=self.root_dir, remove_init=True)
        self._full_module: str = _module_file_to_str(file, root_dir=self.root_dir, remove_init=False)
        # load imports
        imports, scoped_imports = self._parse_imports(file)
        self._imports: List[Import] = imports
        self._scoped_imports: List[Import] = scoped_imports

    @property
    def file(self) -> str: return self._file
    @property
    def root_dir(self) -> str: return self._root_dir
    @property
    def module(self) -> str: return self._module
    @property
    def full_module(self) -> str: return self._full_module
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
            module = '.'.join(_module_file_to_str(file, root_dir=self.root_dir, remove_init=False).split('.')[:-1] + [module])
        return Import(module=module, name=name)

    @staticmethod
    def _ast_in_module_scope(node):
        for parent in ast_iter_parents(node):
            if isinstance(parent, (ast.AsyncFunctionDef, ast.FunctionDef)):
                return False
        return True


class Modules(object):

    def __init__(self, modules):
        self._modules: Dict[str, Module] = {m.module: m for m in modules}
        self._graph: nx.DiGraph = self._make_graph(include_scoped=False)

    def __repr__(self):
        roots = set(m.root_dir for m in self._modules.values())
        return f'Modules<num={len(self._modules)}, roots={sorted(roots)}>'

    @property
    def graph_imported(self):
        return self._graph.copy()

    @property
    def graph_importers(self):
        return self._graph.reverse()

    @property
    def modules(self):
        return list(self._modules.values())

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

    def _make_graph(self, include_scoped=True):
        G = nx.DiGraph()
        for module in self._modules.values():
            if module.full_module == 'disent.frameworks.vae.experimental._weak__st_betavae':
                print(module.imports)
            for imp in module.imports:
                G.add_edge(imp.module, module.module)
            if include_scoped:
                for imp in module.scoped_imports:
                    G.add_edge(imp.module, module.module)
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


def filtered_graph_edges(graph: nx.DiGraph, starts_with):
    graph, removed = graph.copy(), []
    for module, imp in list(graph.edges):
        if not module.startswith(starts_with) or not imp.startswith(starts_with):
            graph.remove_edge(module, imp)
            removed.append((module, imp))
    return graph, removed


# ========================================================================= #
# Loader                                                                    #
# ========================================================================= #


if __name__ == '__main__':
    with TempNumpySeed(777):
        PROJECT_ROOT = os.path.abspath(os.path.join(__file__, '../..'))
        DISENT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, 'disent'))
        modules = Modules.from_dirs(DISENT_ROOT, root_dir=PROJECT_ROOT)

        g_imported, removed = filtered_graph_edges(modules.graph_imported, 'disent.frameworks')

        # visualise
        # Modules.graph_save_visualisation(_modules._graph, os.path.join(PROJECT_ROOT, 'tests/example.html'))

        # for node in sorted(g_imported.nodes):
        #     print(node)
        #     for module, imp, _ in nx.edge_dfs(g_imported, node, orientation='original'):
        #         print('--', module, '->', imp)
        #         print(list(g_imported.neighbors(module)))

        # print cycles
        # print('cycles:')
        # for cycle in sorted(Modules.graph_find_cycles(modules.graph_imported)):
        #     print('--', ' -> '.join(cycle))

        # imports = defaultdict(list)
        # for imp, module in nx.edge_dfs(modules.graph_imported, 'disent.frameworks.framework'):
        #     imp, module = (modules._modules[m].full_module for m in (imp, module))
        #     imports[module].append(imp)
        #     print('>>', module, '->', imp)
        #
        # for module, imports in imports.items():
        #     for imp in imports:
        #         print(module, '->', imp)


