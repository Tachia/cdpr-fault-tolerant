#!/bin/sh
# Builds main.pdf (pdflatex, bibtex, two more passes). The class files of the Elsevier CAS bundle are copied here.
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex > build.log 2>&1
grep -E "^!|undefined|Overfull" main.log | sort | uniq -c
