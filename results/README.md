# Results


This directory contains only this `README.md` file until a script is executed.

When a script runs:
- **CSV tables** (`results/*.csv`): A metrics table named after the experiment is saved directly here (one row per evaluation window).
- **Figures** (`results/figures/*.png`): The script automatically creates the `figures/` subfolder at runtime and saves its generated plots there.

All generated CSVs and PNGs are listed in `.gitignore` and are not committed, as they represent reproducible run artifacts. If you want to version-control specific results or figures, remove their path from `.gitignore` or copy them to the root README.

Every script writes here, whether executed locally or on Colab.
