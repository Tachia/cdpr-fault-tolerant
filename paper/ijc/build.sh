#!/bin/sh
# Builds the two versions of the manuscript for the International Journal of Control:
#   main_anonymous.pdf  for double anonymous review (no author details, funding or repository address)
#   main_full.pdf       with the author details, funding and the repository address
# pdflatex and BibTeX; interact.cls and tfcad.bst are in this folder.
cd "$(dirname "$0")"
build() {   # $1 job name, $2 TeX input
  pdflatex -interaction=nonstopmode -jobname="$1" "$2" >/dev/null
  bibtex "$1" >/dev/null
  pdflatex -interaction=nonstopmode -jobname="$1" "$2" >/dev/null
  pdflatex -interaction=nonstopmode -jobname="$1" "$2" >/dev/null
  echo "$1.pdf: $(pdfinfo "$1.pdf" | grep Pages)"
  grep -E "^!|undefined|Overfull" "$1.log" | sort | uniq -c
}
build main_anonymous main.tex
build main_full '\def\fullversion{}\input{main.tex}'
