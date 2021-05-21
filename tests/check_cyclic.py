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
from functools import lru_cache
from glob import glob
from typing import List
from typing import Optional
from typing import Tuple

import networkx as nx
from matplotlib import pyplot as plt


# ========================================================================= #
# Ast                                                                       #
# ========================================================================= #


def ast_parse(file):
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


def iter_parents(node):
    while hasattr(node, 'parent') and (node.parent is not None):
        node = node.parent
        yield node


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


@dataclasses.dataclass
class Import(object):
    module: str
    name: Optional[str]


@dataclasses.dataclass
class Module(object):
    module: str
    full_module: str
    file: str
    imports: List[Import]
    scoped_imports: List[Import]


# ========================================================================= #
# Module & Import Loader                                                    #
# ========================================================================= #


def in_module_scope(node):
    for parent in iter_parents(node):
        if isinstance(parent, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return False
    return True


@lru_cache()
def file_to_module_str(file, remove_init=False):
    assert file.endswith('.py')
    module = file.replace('/', '.')[:-len('.py')]
    # assert all(str.isidentifier(token) for token in module.split('.'))
    # remove __init__
    if remove_init:
        if module.endswith('.__init__'):
            module = module[:-len('.__init__')]
    return module


def make_from_import(file, data, node, module, name):
    # get line
    line = re.sub(' +', ' ', data[node.lineno - 1].strip())
    assert line.startswith('from ')
    line = line[len('from '):]
    assert line[0] not in string.whitespace
    # check if relative
    if line[0] == '.':
        module = '.'.join(file_to_module_str(file, remove_init=False).split('.')[:-1] + [module])
    return Import(module=module, name=name)


def parse_imports(file) -> Tuple[List[Import], List[Import]]:
    imports, scoped = [], []
    # parse
    root, data = ast_parse(file)
    # visitor
    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for name in node.names:
                self._add(node, Import(module=name.name, name=None))
        def visit_ImportFrom(self, node):
            for name in node.names:
                self._add(node, make_from_import(file=file, data=data, node=node, module=node.module, name=name.name))
        def _add(self, node, imp):
            (imports if in_module_scope(node) else scoped).append(imp)
    # visit
    Visitor().visit(root)
    # done
    return imports, scoped


def parse_module(file) -> Module:
    imports, scoped_imports = parse_imports(file)
    # make module
    return Module(
        module=file_to_module_str(file, remove_init=True),
        full_module=file_to_module_str(file, remove_init=False),
        file=file,
        imports=imports,
        scoped_imports=scoped_imports,
    )


def glob_parse_modules(glob_path='**/*.py', recursive=True):
    module_imports = {}
    for file in glob(glob_path, recursive=recursive):
        module = parse_module(file)
        module_imports[module.module] = module
    return module_imports


# ========================================================================= #
# Create Network                                                            #
# ========================================================================= #


def create_module_graph(modules):
    # create graph
    G = nx.DiGraph()
    for module in modules:
        for imp in module.imports:
            G.add_edge(imp.module, module.module)
        for imp in module.scoped_imports:
            G.add_edge(imp.module, module.module)
    return G


def glob_module_graph(glob_path='**/*.py', recursive=True):
    modules = glob_parse_modules(glob_path=glob_path, recursive=recursive)
    return create_module_graph(modules.values())


def glob_find_cycles(glob_path='**/*.py', recursive=True):
    G = glob_module_graph(glob_path=glob_path, recursive=recursive)
    return list(nx.simple_cycles(G))


def glob_plot(glob_path='**/*.py', recursive=True):
    G = glob_module_graph(glob_path=glob_path, recursive=recursive)
    # plot
    fig, ax = plt.subplots(figsize=(15, 15))
    nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold', ax=ax)
    plt.show()


def glob_visualise(save_file: str, glob_path='**/*.py', recursive=True):
    G = glob_module_graph(glob_path=glob_path, recursive=recursive)
    # check
    assert save_file.endswith('.html')
    # visualise
    from pyvis.network import Network
    net = Network(notebook=False, directed=True, width='100%', height='100%')
    net.from_nx(G)
    net.show(save_file)


# ========================================================================= #
# Loader                                                                    #
# ========================================================================= #


if __name__ == '__main__':
    for cycle in glob_find_cycles(glob_path='**/*.py'):
        print(cycle)
    glob_visualise(save_file=os.path.join(os.path.dirname(__file__), "example.html"))


