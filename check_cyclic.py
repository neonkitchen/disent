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
import re
import string
from collections import namedtuple
from functools import lru_cache
from glob import glob
from typing import List
from typing import Optional
from typing import Tuple


# ========================================================================= #
# Ast                                                                       #
# ========================================================================= #
from matplotlib import pyplot as plt


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
# Loader                                                                    #
# ========================================================================= #



if __name__ == '__main__':

    modules = glob_parse_modules('**/*.py')

    import networkx as nx

    def can_show_module(name):
        return (name.startswith('disent.frameworks.ae') or name.startswith('disent.frameworks.vae')) # and 'experimental' not in name
    def can_show_import(name):
        return (name.startswith('disent.frameworks.ae') or name.startswith('disent.frameworks.vae')) # and 'experimental' not in name

    G = nx.DiGraph()
    for module in modules.values():
        # G.add_node(module.module)
        for imp in module.imports:
            # G.add_node(imp.module)
            if can_show_module(module.module) and can_show_import(imp.module):
                G.add_edge(imp.module, module.module)

    fig, ax = plt.subplots(figsize=(15, 15))
    nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold', ax=ax)
    plt.show()





