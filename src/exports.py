"""Downloadable artifacts and deterministic submission ZIP packaging."""
from pathlib import Path
import json
import shutil
import zipfile


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def make_submission_zip(project_root, output, roll_number):
    if not roll_number or not roll_number.strip():
        raise ValueError("Roll number is required for submission packaging.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in Path(project_root).rglob("*"):
            if path.is_file() and ".venv" not in path.parts and "__pycache__" not in path.parts:
                archive.write(path, Path(project_root).name / path.relative_to(project_root))
        archive.writestr(Path(project_root).name / "ROLL_NUMBER.txt", roll_number.strip())
    return output
