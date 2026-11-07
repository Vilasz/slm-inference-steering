"""Read versioned EvalPlus contents on POSIX without generating or checking candidates."""
import ast
import hashlib
import importlib.metadata
import json

from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
from evalplus.data.mbpp import MBPP_PLUS_VERSION

from evalplus_worker import dataset_json_default


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


class NormalizeNames(ast.NodeTransformer):
    def __init__(self, tree):
        self.names = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.names.setdefault(node.name, f'function_{len(self.names)}')
            if isinstance(node, ast.arg):
                self.names.setdefault(node.arg, f'variable_{len(self.names)}')
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.names.setdefault(node.id, f'variable_{len(self.names)}')

    def visit_FunctionDef(self, node):
        node.name = self.names[node.name]
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Name(self, node):
        node.id = self.names.get(node.id, node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.names.get(node.arg, node.arg)
        return self.generic_visit(node)


if __name__ == '__main__':
    problems = get_mbpp_plus()
    rows = []
    for task, problem in problems.items():
        code = problem['prompt'] + problem['canonical_solution']
        try:
            tree = NormalizeNames(ast.parse(code)).visit(ast.parse(code))
            structural = sha(ast.dump(tree, include_attributes=False))
        except SyntaxError:
            structural = None
        rows.append({'task_id': task, 'prompt': problem['prompt'], 'prompt_sha256': sha(problem['prompt']),
                     'canonical_structure_sha256': structural,
                     'base_inputs_sha256': sha(json.dumps(problem['base_input'], sort_keys=True, default=dataset_json_default)),
                     'plus_inputs_sha256': sha(json.dumps(problem['plus_input'], sort_keys=True, default=dataset_json_default))})
    print(json.dumps({'metadata': {'version': MBPP_PLUS_VERSION, 'evalplus': importlib.metadata.version('evalplus'),
                                  'official_md5': get_mbpp_plus_hash(),
                                  'dataset_sha256': sha(json.dumps(problems, sort_keys=True, default=dataset_json_default))},
                      'tasks': rows}, ensure_ascii=True))
