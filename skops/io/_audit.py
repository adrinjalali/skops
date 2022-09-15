import json
from zipfile import ZipFile


def _is_unsafe(node):
    try:
        _, _ = node["__module__"], node["__class__"]
    except Exception:
        # can't figure type, so nothing's run and it's not unsafe.
        return False

    if node["__class__"] == "function" or node["__class__"] == "partial":
        return True
    if node["__module__"].split(".")[0] not in (
        "builtins",
        "numpy",
        "scipy",
        "sklearn",
    ):
        return True
    return False


def _audit_node(node):
    """Recursively audit a node in the schema."""
    unsafe = []
    if _is_unsafe(node):
        unsafe.append(node)

    if isinstance(node, dict):
        for key, value in node.items():
            unsafe.extend(_audit_node(value))
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict):
                unsafe.extend(_audit_node(item))
    return unsafe


def audit(file):
    input_zip = ZipFile(file)
    schema = input_zip.read("schema.json")
    schema = json.loads(schema)
    unsafe = _audit_node(schema)
    if unsafe:
        raise Exception("The following nodes are unsafe: " + str(unsafe))
