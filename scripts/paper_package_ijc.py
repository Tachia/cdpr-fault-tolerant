"""Assemble the LaTeX source of the version for the International Journal of Control into one zip.

The journal reviews double anonymously and asks for two PDF files, "Manuscript - anonymous" and
"Manuscript - with author details", and for the LaTeX source as a single zip file ("LaTeX Source
Files"). The zip follows the layout of the Taylor & Francis Interact template: one source file,
main.tex, with the tables written into it, its bibliography (main.bbl and references.bib), the class
interact.cls, the style tfcad.bst and the figures as vector PDF. main.tex gives the anonymous version;
\\blindfalse in place of \\blindtrue gives the version with the author details. The script compiles both
versions from that file in a temporary folder, fails if either does not compile cleanly, and writes
the two PDF files from the same source.
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
HEADER = re.compile(r"\A(%[^\n]*\n)+")
SWITCH = "\\newif\\ifblind\n\\blindtrue\n\\ifdefined\\fullversion\\blindfalse\\fi\n"
SOURCE_HEADER = ("% Fault tolerant control of a redundant cable driven parallel robot\n"
                 "% pdflatex, bibtex, pdflatex, pdflatex. Anonymous version: \\blindtrue;\n"
                 "% version with the author details: \\blindfalse.\n")


def source() -> str:
    """main.tex of the repository with the tables written in and the build switch made plain."""
    tex = (MS / "main.tex").read_text()
    assert SWITCH in tex, "the version switch is missing from paper/ijc/main.tex"
    tex = HEADER.sub(lambda m: SOURCE_HEADER, tex, count=1)
    tex = tex.replace(SWITCH, "\\newif\\ifblind\n\\blindtrue\n")

    def table(m):
        body = (MS / "tables" / f"{m.group(1)}.tex").read_text()
        return "".join(l for l in body.splitlines(keepends=True) if not l.startswith("% written by")).rstrip("\n")

    tex = re.sub(r"\\input\{tables/([^}]+)\}", table, tex)
    assert "\\input{" not in tex
    return tex


def compile_tex(folder: Path, name: str) -> None:
    for cmd in (["pdflatex", "-interaction=nonstopmode", name], ["bibtex", name],
                ["pdflatex", "-interaction=nonstopmode", name], ["pdflatex", "-interaction=nonstopmode", name]):
        subprocess.run(cmd, cwd=folder, capture_output=True, check=False)
    log = (folder / f"{name}.log").read_text(errors="replace")
    if re.search(r"^! |Citation .* undefined|Reference .* undefined|Overfull", log, re.M):
        print(f"{name} does not compile cleanly; see its log in {folder}")
        sys.exit(1)


def main():
    tex = source()
    figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
    with tempfile.TemporaryDirectory() as tmp:
        pkg, full = Path(tmp) / "pkg", Path(tmp) / "full"
        for d in (pkg, full):
            (d / "figures").mkdir(parents=True)
            for f in ("references.bib", "interact.cls", "tfcad.bst"):
                shutil.copy2(MS / f, d / f)
            for f in figures:
                shutil.copy2(MS / "figures" / f"{f}.pdf", d / "figures" / f"{f}.pdf")
        (pkg / "main.tex").write_text(tex)
        (full / "main.tex").write_text(tex.replace("\\newif\\ifblind\n\\blindtrue\n", "\\newif\\ifblind\n\\blindfalse\n"))
        compile_tex(pkg, "main")
        compile_tex(full, "main")
        assert (pkg / "main.bbl").read_text() == (full / "main.bbl").read_text()
        shutil.copy2(pkg / "main.pdf", MS / "main_anonymous.pdf")
        shutil.copy2(full / "main.pdf", MS / "main_full.pdf")
        files = ["main.tex", "main.bbl", "references.bib", "interact.cls", "tfcad.bst"] + \
                [f"figures/{f}.pdf" for f in figures]
        out = MS / f"{NAME}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(pkg / f, f)
    print(f"{out.relative_to(ROOT)}: {len(files)} files ({len(figures)} figures); "
          f"main_anonymous.pdf and main_full.pdf written from the same source")


if __name__ == "__main__":
    main()
