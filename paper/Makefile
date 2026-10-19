PYTHON ?= python
LATEXMK ?= latexmk

note.pdf: note.tex refs.bib figures/summary.pdf figures/budget_vs_time.pdf figures/gap_mechanisms.pdf
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error note.tex

figures/summary.pdf figures/budget_vs_time.pdf figures/gap_mechanisms.pdf: make_figures.py ../figures/comparison.json
	$(PYTHON) make_figures.py

clean:
	$(LATEXMK) -c note.tex
	rm -f noteNotes.bib

.PHONY: clean
