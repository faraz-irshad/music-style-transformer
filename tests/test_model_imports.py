"""Static checks for names imported from project model modules."""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _assignment_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_assignment_names(element))
        return names
    return set()


def _top_level_definitions(module: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_assignment_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_assignment_names(node.target))
    return names


def _target_module_path(module_name: str) -> Path:
    module_path = ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
    if module_path.is_file():
        return module_path
    package_path = ROOT.joinpath(*module_name.split(".")) / "__init__.py"
    return package_path


def test_model_import_names_exist_at_module_top_level():
    failures: list[str] = []
    for package in (ROOT / "model", ROOT / "controller"):
        for source_path in sorted(package.rglob("*.py")):
            source = ast.parse(source_path.read_text(), filename=str(source_path))
            for node in ast.walk(source):
                if not (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith("model.")
                ):
                    continue

                target_path = _target_module_path(node.module)
                if not target_path.is_file():
                    failures.append(
                        f"{source_path.relative_to(ROOT)}:{node.lineno}: "
                        f"cannot find imported module {node.module}"
                    )
                    continue

                target = ast.parse(
                    target_path.read_text(), filename=str(target_path)
                )
                defined_names = _top_level_definitions(target)
                for alias in node.names:
                    if alias.name != "*" and alias.name not in defined_names:
                        failures.append(
                            f"{source_path.relative_to(ROOT)}:{node.lineno}: "
                            f"{node.module}.{alias.name} is not defined at "
                            f"top level in {target_path.relative_to(ROOT)}"
                        )

    assert not failures, "\n".join(failures)
