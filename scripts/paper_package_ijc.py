"""Assemble the LaTeX source of the version for the International Journal of Control into one zip.

The journal reviews double anonymously and asks for two PDF files, "Manuscript - anonymous" and
"Manuscript - with author details", and for the LaTeX source as a single zip file ("LaTeX Source
Files"). Both versions come from paper/ijc/main.tex; in the package each is a file of its own, so that
either compiles at the top level with pdflatex and BibTeX (or with latex and dvips, from the EPS
copies of the figures). The script compiles both in a temporary folder, keeps their .bbl files, and
fails if either does not compile cleanly.
Output: paper/ijc/ijc_latex_source.zip, paper/ijc/main_anonymous.pdf, paper/ijc/main_full.pdf
Usage: python scripts/paper_package_ijc.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "paper" / "ijc"
NAME = "ijc_latex_source"
SWITCH = "\\newif\\ifblind\n\\blindtrue\n\\ifdefined\\fullversion\\blindfalse\\fi\n"

README = """Fault tolerant control of a redundant cable driven parallel robot with model based fault isolation
and precomputed recovery plans

LaTeX source for the International Journal of Control (Taylor & Francis Interact layout, author-date
references with tfcad.bst).

  main_anonymous.tex    the anonymous version for double anonymous review (no author details,
                        funding, contributions or repository address)
  main_full.tex         the version with the author details
  main_*.bbl            the bibliographies as BibTeX wrote them
  references.bib        the bibliography database
  interact.cls, tfcad.bst   Taylor & Francis template files
  figures/              the eleven figures, vector PDF (pdflatex) and EPS (latex and dvips)
  tables/               the tables of parameters, methods, results and the latency study

Compile either version: pdflatex <file>, bibtex <file>, pdflatex <file>, pdflatex <file>.
The two files differ only in the switch at the top (\\blindtrue or \\blindfalse).
"""


def run(cmds, cwd):
    for cmd in cmds:
        subprocess.run(cmd, cwd=cwd, capture_output=True, check=False)


def main():
    tex = (MS / "main.tex").read_text()
    assert SWITCH in tex, "the version switch is missing from paper/ijc/main.tex"
    versions = {"main_anonymous": tex.replace(SWITCH, "\\newif\\ifblind\n\\blindtrue\n"),
                "main_full": tex.replace(SWITCH, "\\newif\\ifblind\n\\blindfalse\n")}
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / NAME
        (pkg / "figures").mkdir(parents=True)
        (pkg / "tables").mkdir()
        for f in ("references.bib", "interact.cls", "tfcad.bst"):
            shutil.copy2(MS / f, pkg / f)
        figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
        for f in figures:
            shutil.copy2(MS / "figures" / f"{f}.pdf", pkg / "figures" / f"{f}.pdf")
            subprocess.run(["pdftops", "-eps", str(pkg / "figures" / f"{f}.pdf"), str(pkg / "figures" / f"{f}.eps")],
                           check=True)
        tables = sorted(set(re.findall(r"\\input\{tables/([^}]+)\}", tex)))
        for t in tables:
            shutil.copy2(MS / "tables" / f"{t}.tex", pkg / "tables" / f"{t}.tex")
        for name, body in versions.items():
            (pkg / f"{name}.tex").write_text(body)
            run([["pdflatex", "-interaction=nonstopmode", name], ["bibtex", name],
                 ["pdflatex", "-interaction=nonstopmode", name], ["pdflatex", "-interaction=nonstopmode", name]], pkg)
            log = (pkg / f"{name}.log").read_text(errors="replace")
            if re.search(r"^! |Citation .* undefined|Reference .* undefined", log, re.M):
                print(f"{name} does not compile cleanly; see its log")
                sys.exit(1)
            shutil.copy2(pkg / f"{name}.pdf", MS / f"{name}.pdf")
        (pkg / "README.txt").write_text(README)
        keep = {"references.bib", "interact.cls", "tfcad.bst", "README.txt"} | {f"{n}{e}" for n in versions
                                                                              for e in (".tex", ".bbl")}
        for f in pkg.iterdir():
            if f.is_file() and f.name not in keep:
                f.unlink()
        out = MS / f"{NAME}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(pkg.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(pkg.parent))
        print(f"{out.relative_to(ROOT)}: {len(figures)} figures, {len(tables)} tables, "
              f"{len(zipfile.ZipFile(out).namelist())} files")


if __name__ == "__main__":
    main()
