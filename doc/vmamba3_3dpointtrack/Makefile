TEX  = pdflatex
BIB  = bibtex
SRC  = vmamba3_3dpointtrack.tex
BASE = $(SRC:.tex=)
PDF  = $(BASE).pdf
# Figures are prerequisites. Without them, swapping a regenerated chart into figs/ and running
# make is a SILENT NO-OP -- the .tex is untouched, so make sees the PDF as up to date and the
# document keeps rendering the old image. That happened on 2026-08-15 and cost a round of
# debugging a legend fix that had in fact already worked.
FIGS = $(wildcard figs/*.pdf figs/*.png figs/*.jpg)

.PHONY: all clean

all: $(PDF)

$(PDF): $(SRC) refs.bib $(FIGS)
	$(TEX) $(SRC)
	$(BIB) $(BASE)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz *.fls *.fdb_latexmk *.bbl *.blg *.pdf
