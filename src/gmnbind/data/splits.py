"""Train/val/test splits stored as plain text files (one PDB code per line)."""
from pathlib import Path


def read_codes(path):
    """Read a text file with one PDB code per line (empty lines are ignored)."""
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_codes(path, codes):
    """Write PDB codes to a text file, one per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(codes) + "\n", encoding="utf-8")


def load_split(split_dir):
    """Read train.txt, val.txt and (optional) test.txt -> {"train": [...], "val": [...], "test": [...]}."""
    split_dir = Path(split_dir)
    split = {"train": read_codes(split_dir / "train.txt"),
             "val":   read_codes(split_dir / "val.txt")}
    if (split_dir / "test.txt").exists():
        split["test"] = read_codes(split_dir / "test.txt")
    return split