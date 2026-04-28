#!/usr/bin/env python3
"""Apply `constant_replace` edits to a Python source file.

Reads a JSON request on stdin of the form

    {"src": "<file contents>",
     "edits": [{"name": "ADAM_BETAS",
                "expected_old_repr": "(0.9, 0.95)",
                "new_repr": "(0.9, 0.97)"}, ...]}

and writes a JSON response on stdout:

    {"ok": true, "new_src": "..."}                # success
    {"ok": false, "reason": "...", "index": 0}    # first failing edit

Match semantics:
  * Only module-level Assign / AnnAssign targets are considered.
  * The current value is compared to expected_old_repr after AST-normalizing
    both via `ast.unparse(ast.parse(..., mode='eval'))` so spacing differences
    don't matter. As a fallback, both expressions are evaluated in a
    builtins-stripped scope; if they produce equal Python objects the edit is
    accepted (`524288` matches `2 ** 19` matches `1 << 19`).
  * The replacement is a textual splice on the original source so surrounding
    formatting and comments are preserved exactly.
  * After all edits are applied, the result is re-parsed; a syntax error
    aborts the whole batch (no half-applied edits are written back).
"""

import ast
import json
import sys
from typing import Any, Tuple


def _normalize_expr(text: str) -> str:
    """Round-trip an expression through `ast.unparse` so trivial spacing /
    parenthesization differences disappear. Raises `SyntaxError` if `text`
    isn't a parseable expression."""
    return ast.unparse(ast.parse(text, mode='eval'))


def _safe_eval(text: str) -> Tuple[bool, Any]:
    """Evaluate `text` in a builtins-free scope; returns (ok, value)."""
    try:
        value = eval(compile(text, '<edit>', 'eval'), {'__builtins__': {}}, {})
        return True, value
    except Exception:
        return False, None


def _values_match(cur_src: str, expected_repr: str) -> bool:
    try:
        if _normalize_expr(cur_src) == _normalize_expr(expected_repr):
            return True
    except SyntaxError:
        return False
    cur_ok, cur_val = _safe_eval(cur_src)
    exp_ok, exp_val = _safe_eval(expected_repr)
    return cur_ok and exp_ok and cur_val == exp_val


def _find_module_assignment(tree: ast.Module, name: str):
    """Find the first module-level assignment whose target is `name`. Returns
    the `ast.AST` node holding the value, or `None`."""
    for top in tree.body:
        if (
            isinstance(top, ast.Assign)
            and len(top.targets) == 1
            and isinstance(top.targets[0], ast.Name)
            and top.targets[0].id == name
        ):
            return top.value
        if (
            isinstance(top, ast.AnnAssign)
            and isinstance(top.target, ast.Name)
            and top.target.id == name
            and top.value is not None
        ):
            return top.value
    return None


def _offset_of(src: str, line: int, col: int) -> int:
    """Convert 1-indexed line + 0-indexed col to a flat character offset."""
    lines = src.split('\n')
    return sum(len(line_text) + 1 for line_text in lines[: line - 1]) + col


def main() -> int:
    request = json.loads(sys.stdin.read())
    src: str = request['src']
    edits = request['edits']

    tree = ast.parse(src)

    splices = []
    for i, edit in enumerate(edits):
        name = edit['name']
        expected_repr = edit['expected_old_repr']
        new_repr = edit['new_repr']

        value_node = _find_module_assignment(tree, name)
        if value_node is None:
            print(json.dumps({
                'ok': False, 'index': i,
                'reason': f'no module-level assignment to {name!r}',
            }))
            return 0

        cur_src = ast.unparse(value_node)
        if not _values_match(cur_src, expected_repr):
            print(json.dumps({
                'ok': False, 'index': i,
                'reason': (
                    f'current value of {name} ({cur_src!r}) does not match '
                    f'expected_old_repr ({expected_repr!r})'
                ),
            }))
            return 0

        try:
            ast.parse(new_repr, mode='eval')
        except SyntaxError as exc:
            print(json.dumps({
                'ok': False, 'index': i,
                'reason': f'new_repr {new_repr!r} is not a parseable expression: {exc}',
            }))
            return 0

        start = _offset_of(src, value_node.lineno, value_node.col_offset)
        end = _offset_of(src, value_node.end_lineno, value_node.end_col_offset)
        splices.append((start, end, new_repr))

    new_src = src
    for start, end, replacement in sorted(splices, key=lambda x: -x[0]):
        new_src = new_src[:start] + replacement + new_src[end:]

    try:
        ast.parse(new_src)
    except SyntaxError as exc:
        print(json.dumps({
            'ok': False,
            'reason': f'patched file does not parse: {exc}',
        }))
        return 0

    print(json.dumps({'ok': True, 'new_src': new_src}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
