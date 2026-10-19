# Note: error budgets for four published silicon devices

`note.tex` is a short technical note on the device comparison in the main README: how much of the measured two-qubit gate error of Xue 2022, Noiri 2022, Mills 2022 and Tanttu 2024 (device A) the characterised noise accounts for, and which further mechanisms could close the gap. `note.pdf` is the built version.

Build (needs a TeX distribution with `revtex4-2`, `newtx` and `latexmk`, and matplotlib):

```bash
cd paper
make                      # draws the figures if needed, then runs latexmk
make PYTHON=/path/to/python
make clean                # removes the LaTeX build files, keeps note.pdf
```

`make_figures.py` draws the three figures as PDF from `../figures/comparison.json` and recomputes nothing. To regenerate that file from the package, run `python benchmarks/make_figures.py` from the repository root first (about five minutes).

Every number in the note is taken from `figures/comparison.json`, `src/spin_optimal_control/data/devices.json` or the main README, which are produced by the package.
