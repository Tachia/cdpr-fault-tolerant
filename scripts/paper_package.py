"""Assemble the LaTeX source of the manuscript into one folder and zip it for the journal or Overleaf.

The package holds main.tex, references.bib, the class files of the Elsevier CAS template, the figures
(PDF), the tables and a copy of the compiled PDF, laid out so that main.tex compiles at the top level
with pdflatex and BibTeX. The script compiles the package in a temporary folder before writing the zip.
Output: paper/manuscript/cjme_manuscript.zip
Usage: python scripts/paper_package.py
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
MS = ROOT / "paper" / "manuscript"
NAME = "cjme_manuscript"

README = """Fault tolerant control of a redundant cable driven parallel robot with model based fault isolation
and precomputed recovery plans

LaTeX source for the Chinese Journal of Mechanical Engineering (Elsevier CAS double column template).

Compile: pdflatex main, bibtex main, pdflatex main, pdflatex main
(on Overleaf: upload this zip as a new project; the compiler is pdfLaTeX and the main file main.tex).

  main.tex                  manuscript
  references.bib            bibliography (style elsarticle-num-names.bst)
  cas-dc.cls, cas-common.sty, elsarticle-num-names.bst, thumbnails/   Elsevier CAS template files
  figures/                  the eleven figures (vector PDF)
  tables/                   the tables of parameters, methods and results
  highlights.txt            the highlights, as a separate file for the submission system
  main.pdf                  the compiled manuscript
"""


def highlights(tex: str) -> str:
    block = tex[tex.index("\\begin{highlights}"):tex.index("\\end{highlights}")]
    return "\n".join(line.replace("\\item", "").strip() for line in block.splitlines() if "\\item" in line) + "\n"


def main():
    tex = (MS / "main.tex").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / NAME
        (pkg / "figures").mkdir(parents=True)
        (pkg / "tables").mkdir()
        for f in ("main.tex", "references.bib", "cas-dc.cls", "cas-common.sty", "elsarticle-num-names.bst"):
            shutil.copy2(MS / f, pkg / f)
        shutil.copytree(MS / "thumbnails", pkg / "thumbnails")
        figures = sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)))
        for f in figures:
            shutil.copy2(ROOT / "paper" / "figures" / f, pkg / "figures" / f)
        tables = sorted(set(re.findall(r"\\tablefile\{([^}]+)\}", tex)))
        for t in tables:
            shutil.copy2(ROOT / "paper" / "tables" / f"{t}.tex", pkg / "tables" / f"{t}.tex")
        (pkg / "highlights.txt").write_text(highlights(tex))
        (pkg / "README.txt").write_text(README)
        for cmd in (["pdflatex", "-interaction=nonstopmode", "main"], ["bibtex", "main"],
                    ["pdflatex", "-interaction=nonstopmode", "main"], ["pdflatex", "-interaction=nonstopmode", "main"]):
            subprocess.run(cmd, cwd=pkg, capture_output=True, check=False)
        log = (pkg / "main.log").read_text(errors="replace")
        if re.search(r"^! |Citation .* undefined|Reference .* undefined", log, re.M):
            print("the package does not compile cleanly; see main.log")
            sys.exit(1)
        keep = {"main.tex", "references.bib", "cas-dc.cls", "cas-common.sty", "elsarticle-num-names.bst",
                "main.pdf", "highlights.txt", "README.txt"}
        for f in pkg.iterdir():
            if f.is_file() and f.name not in keep:
                f.unlink()
        out = MS / f"{NAME}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(pkg.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(pkg.parent))
        print(f"{out.relative_to(ROOT)}: {len(figures)} figures, {len(tables)} tables, "
              f"{sum(1 for _ in zipfile.ZipFile(out).namelist())} files")


if __name__ == "__main__":
    main()
