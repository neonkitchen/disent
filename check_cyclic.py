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
from collections import namedtuple
from glob import glob
from typing import List
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
    return root


def iter_parents(node):
    while hasattr(node, 'parent') and (node.parent is not None):
        node = node.parent
        yield node


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


Import = namedtuple('Import', 'module name')
Module = namedtuple('Module', 'module full_module file imports scoped_imports')


# ========================================================================= #
# Module & Import Loader                                                    #
# ========================================================================= #


def in_module_scope(node):
    for parent in iter_parents(node):
        if isinstance(parent, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return False
    return True


def parse_imports(file) -> Tuple[List[Import], List[Import]]:
    imports, scoped = [], []
    # visitor
    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for name in node.names:
                self._add(node, Import(module=name.name, name=None))
        def visit_ImportFrom(self, node):
            for name in node.names:
                self._add(node, Import(module=node.module, name=name.name))
        def _add(self, node, imp):
            (imports if in_module_scope(node) else scoped).append(imp)
    # visit
    Visitor().visit(ast_parse(file))
    # done
    return imports, scoped


def parse_module(file) -> Module:
    imports, scoped_imports = parse_imports(file)
    # get module strings
    full_module = file.replace('/', '.')[:-len('.py')]
    module = full_module[:-len('.__init__')] if full_module.endswith('.__init__') else full_module
    # make module
    return Module(
        module=module,
        full_module=full_module,
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

    modules = glob_parse_modules('disent/frameworks/ae/**/*.py')

    import networkx as nx

    G = nx.Graph()
    for module in modules.values():
        print(module.module)
        G.add_node(module.module)
        for imp in module.imports:
            print('    ', imp.module)
            G.add_edge(module.module, imp.module)
    H = nx.DiGraph(G)

    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()





