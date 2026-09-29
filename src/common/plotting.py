"""Make plt.show() also save a PNG, so figures are not lost outside a live kernel.

plt.show() renders inline when this code runs inside a notebook's own
interactive kernel (Colab, Jupyter), which is how the original notebooks
were written. It silently does nothing when a script runs as a subprocess,
for example via `!python script.py`, which is the normal way to run these
scripts, because there is no interactive kernel to render into. Every
plt.show() call in every script is left exactly as it was; this module
patches plt.show once, at the start of each script, so each figure is also
saved as a PNG and nothing is lost either way.
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt

_counter = itertools.count(1)


def enable_figure_autosave(results_directory: Path, prefix: str) -> None:
    figures_dir = results_directory / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    original_show = plt.show

    def show_and_save(*args, **kwargs):
        fig = plt.gcf()
        if fig.get_axes():
            n = next(_counter)
            fig.savefig(figures_dir / f"{prefix}_{n:02d}.png", dpi=150, bbox_inches="tight")
        return original_show(*args, **kwargs)

    plt.show = show_and_save
