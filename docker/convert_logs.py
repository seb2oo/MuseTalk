########################################################
########## DONT USE IT CREATE PROBLEME !!!!#############
########################################################

import ast
import sys
from pathlib import Path


def get_source_offsets(source, node):
    """Return character offsets for an AST node."""
    lines = source.splitlines(keepends=True)

    start = sum(len(line) for line in lines[:node.lineno - 1])
    start += node.col_offset

    end = sum(len(line) for line in lines[:node.end_lineno - 1])
    end += node.end_col_offset

    return start, end


def convert_print_to_debug(file_path):
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")

    tree = ast.parse(source)

    replacements = []

    for node in ast.walk(tree):

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            start, end = get_source_offsets(source, node)

            original = source[start:end]

            replacement = "logger.debug" + original[len("print"):]

            replacements.append((start, end, replacement))

    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]

    path.write_text(source, encoding="utf-8")

    print(f"\n{len(replacements)} print() -> logger.debug()")
    print(f"File: {path}")


def convert_debug_to_print(file_path):
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")

    tree = ast.parse(source)

    replacements = []

    for node in ast.walk(tree):

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "debug"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
        ):
            start, end = get_source_offsets(source, node)

            original = source[start:end]

            replacement = "print" + original[len("logger.debug"):]

            replacements.append((start, end, replacement))

    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]

    path.write_text(source, encoding="utf-8")

    print(f"\n{len(replacements)} logger.debug() -> print()")
    print(f"File: {path}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage:")
        print("  python convert_logs.py debug handler.py")
        print("  python convert_logs.py print handler.py")
        sys.exit(1)

    mode = sys.argv[1]
    file_path = sys.argv[2]

    if mode == "debug":
        convert_print_to_debug(file_path)

    elif mode == "print":
        convert_debug_to_print(file_path)

    else:
        print("ERROR: mode must be 'debug' or 'print'")
        sys.exit(1)

"""
Pour ton handler.py

Tu fais :

python convert_logs.py debug handler.py

Tous tes :

print(...)

deviennent :

logger.debug(...)

Et avant ça, tu ajoutes simplement dans handler.py :

import logging

logger = logging.getLogger("musetalk")

Puis, quand tu veux retrouver ton handler exactement comme avant :

python convert_logs.py print handler.py
Et surtout

Le script ne touche pas à :

logger.info(...)
logger.warning(...)
logger.error(...)
logger.critical(...)

Il ne touche qu'à :

print(...)

et :

logger.debug(...)

Donc tu peux vraiment faire :

handler.py
   ↓
print → logger.debug
   ↓
tests RunPod
   ↓
logger.debug → print
   ↓
handler.py original

C'est exactement adapté à ton besoin : tu gardes ton handler comme source de logs détaillés, 
mais tu peux basculer temporairement tous ces logs vers le système de niveaux de RunPod.
"""