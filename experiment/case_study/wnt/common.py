import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = ("yeast", "multispecies", "multiclass", "humanvirus")
TITLES = ("Yeast", "Multi-species", "Multi-class", "Human-virus")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def setup_style(path):
    import matplotlib.pyplot as plt
    style = read_json(path)
    plt.rcParams.update({"font.family": style["font_family"], "font.size": style["font_size"],
                         "pdf.fonttype": 42, "axes.linewidth": .7})
    return style

def export(fig, stem, style):
    import matplotlib.pyplot as plt
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in style["formats"]:
        if ext not in ("png", "pdf", "tiff"):
            raise ValueError("Only PNG, PDF and TIFF are supported")
        fig.savefig(stem.with_suffix('.' + ext), dpi=style["dpi"], facecolor="white")
    plt.close(fig)
