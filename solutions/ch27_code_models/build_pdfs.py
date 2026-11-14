"""Typeset the executed Chapter 27 notebook as two matching solution PDFs.

Requires mistune and either Tectonic or XeLaTeX on PATH. An explicit engine
path can be supplied with --engine. The written PDF contains every solution;
the notebook PDF adds code and recorded outputs. Run the notebook first.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import mistune


ROOT = Path(__file__).resolve().parent
PARSE = mistune.create_markdown(renderer="ast", plugins=["table", "math"])
SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "’": "'", "‘": "`", "“": "``", "”": "''", "–": "--", "—": "---",
}


def escape(value):
    return "".join(SPECIAL.get(character, character) for character in value)


def inline(tokens):
    result = []
    for token in tokens:
        kind = token["type"]
        if kind == "text":
            result.append(escape(token["raw"]))
        elif kind in ("softbreak", "linebreak"):
            result.append(" " if kind == "softbreak" else r"\newline ")
        elif kind == "inline_math":
            result.append("$" + token["raw"] + "$")
        elif kind == "block_math":
            result.append("\\[\n" + token["raw"] + "\n\\]")
        elif kind == "codespan":
            result.append(r"\texttt{" + escape(token["raw"]) + "}")
        elif kind in ("emphasis", "strong"):
            command = "emph" if kind == "emphasis" else "textbf"
            result.append("\\" + command + "{" + inline(token["children"]) + "}")
        elif kind == "link":
            result.append(inline(token["children"]))
        else:
            raise ValueError(f"Unsupported inline Markdown: {kind}")
    return "".join(result)


def table(token):
    header, body = token["children"]
    rows = [header["children"]] + [row["children"] for row in body["children"]]
    values = [[inline(cell["children"]) for cell in row] for row in rows]
    # p-columns wrap prose and keep mathematical expressions properly typeset.
    weights = [max(12, min(44, max(len(value[i]) for value in values)))
               for i in range(len(values[0]))]
    total = sum(weights)
    columns = "".join(
        r">{\raggedright\arraybackslash}p{\dimexpr"
        + f"{0.99 * weight/total:.5f}" + r"\linewidth-2\tabcolsep\relax}"
        for weight in weights
    )
    result = [r"\par\medskip\noindent{\small\renewcommand{\arraystretch}{1.20}%",
              r"\begin{tabular}{" + columns + "}", r"\toprule"]
    result.append(" & ".join(r"\textbf{" + cell + "}" for cell in values[0]) + r"\\")
    result.append(r"\midrule")
    result.extend(" & ".join(row) + r"\\" for row in values[1:])
    result.extend([r"\bottomrule", r"\end{tabular}}\par\medskip"])
    return "\n".join(result)


def markdown(source):
    result = []
    previous = None
    for token in PARSE(source):
        kind = token["type"]
        if kind == "blank_line":
            continue
        if kind == "heading":
            heading = inline(token["children"])
            result.append(r"\section*{" + heading + "}")
            result.append(r"\addcontentsline{toc}{section}{" + heading + "}")
        elif kind == "paragraph":
            result.append(r"\begin{samepage}" + "\n" + inline(token["children"])
                          + "\n" + r"\par\end{samepage}")
        elif kind == "block_math":
            equation = "\\[\n" + token["raw"] + "\n\\]"
            # Keep the sentence introducing an equation with that equation.
            # A blank paragraph before the display otherwise permits a break.
            if previous == "paragraph":
                result[-1] = (r"\par\noindent\begin{minipage}{\linewidth}" + "\n"
                              + result[-1].rstrip() + "\n" + equation
                              + r"\end{minipage}\par")
            else:
                result.append(equation)
        elif kind == "table":
            if previous == "paragraph":
                result[-1] = (r"\par\noindent\begin{minipage}{\linewidth}" + "\n"
                              + result[-1].rstrip() + "\n" + table(token)
                              + r"\end{minipage}\par")
            else:
                result.append(table(token))
        elif kind == "block_code":
            result.append(listing(token["raw"], "code"))
        else:
            raise ValueError(f"Unsupported Markdown block: {kind}")
        previous = kind
    return "\n\n".join(result)


def listing(source, style):
    if r"\end{lstlisting}" in source:
        raise ValueError("A listing contains a LaTeX environment terminator")
    return (r"\begin{lstlisting}[style=" + style + "]\n"
            + source.rstrip() + "\n" + r"\end{lstlisting}")


def preamble(title):
    return r"""\documentclass[11pt,letterpaper]{article}
\usepackage[T1]{fontenc}
\renewcommand{\rmdefault}{ptm}
\renewcommand{\sfdefault}{phv}
\usepackage[margin=.8in,headheight=14pt,headsep=15pt]{geometry}
\usepackage[cmex10]{amsmath}
\usepackage{amssymb,array,booktabs,xcolor,listings,needspace,graphicx}
\DeclareFontShape{OMX}{cmex}{m}{n}{<->cmex10}{}
\usepackage[hidelinks]{hyperref}
\definecolor{ink}{HTML}{26323D}
\definecolor{blue}{HTML}{3976AF}
\definecolor{gray}{HTML}{687580}
\definecolor{green}{HTML}{46856E}
\definecolor{line}{HTML}{DCE2E7}
\lstdefinestyle{code}{language=Python,
  basicstyle=\ttfamily\fontsize{8.2}{10}\selectfont,
  keywordstyle=\color{blue},commentstyle=\color{gray},stringstyle=\color{green},
  frame=single,framerule=.3pt,framesep=5pt,rulecolor=\color{line},
  backgroundcolor=\color[HTML]{F7F9FA},breaklines=true,breakatwhitespace=true,
  columns=fullflexible,keepspaces=true,showstringspaces=false,
  xleftmargin=6pt,xrightmargin=6pt,aboveskip=8pt,belowskip=10pt}
\lstdefinestyle{output}{style=code,language={},backgroundcolor=\color[HTML]{EFF6F2}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\emergencystretch}{2em}
\setlength{\tabcolsep}{5pt}
\raggedbottom
\widowpenalty=10000
\clubpenalty=10000
\makeatletter
\def\ps@plain{%
\def\@oddhead{\small\sffamily\color{gray}TRANSFORMERS\quad |\quad CHAPTER 27\hfil TITLE}%
\let\@evenhead\@oddhead
\def\@oddfoot{\small\sffamily\color{gray}Models for Code\hfil\thepage}%
\let\@evenfoot\@oddfoot}
\makeatother
\pagestyle{plain}
\hypersetup{pdftitle={Chapter 27: TITLE},pdfauthor={Juan Terven and Diana Córdova},pdflang={en-US}}
\begin{document}
\color{ink}
""".replace("TITLE", title)


def plot_output(output, build_dir):
    data = output.get("data", {})
    if "image/png" not in data:
        return None
    encoded = "".join(data["image/png"])
    raw = base64.b64decode(encoded)
    filename = "notebook-plot-" + hashlib.sha256(raw).hexdigest()[:12] + ".png"
    path = build_dir / filename
    path.write_bytes(raw)
    return (r"\par\medskip\noindent\includegraphics[width=\linewidth]{"
            + filename + r"}\par\medskip")


def document(notebook, include_code, build_dir):
    title = "Worked solutions with Python" if include_code else "Written solutions"
    pieces = [preamble(title)]
    if not include_code:
        pieces[0] = pieces[0].replace(r"\setlength{\parskip}{6pt}", r"\setlength{\parskip}{3pt}")
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            if not include_code and "notebook-only" in cell.get("metadata", {}).get("tags", []):
                continue
            source = source.replace("# Chapter 27 | Worked solutions", "# Chapter 27 | " + title)
            pieces.append(markdown(source))
        elif cell["cell_type"] == "code":
            if not include_code:
                for output in cell.get("outputs", []):
                    figure = plot_output(output, build_dir)
                    if figure:
                        pieces.append(figure)
                continue
            # These short teaching cells fit on one page. Keep each calculation
            # beside its recorded result rather than splitting a few final lines.
            line_count = len(source.splitlines()) + sum(
                len("".join(output.get("text", [])).splitlines())
                for output in cell.get("outputs", []))
            has_plot = any("image/png" in output.get("data", {}) for output in cell.get("outputs", []))
            keep_together = line_count <= 54 and not has_plot
            if keep_together:
                pieces.append(r"\par\medskip\noindent\begin{minipage}{\linewidth}")
            else:
                pieces.append(r"\Needspace{12\baselineskip}")
                if len(source.splitlines()) <= 54:
                    pieces.append(r"\par\noindent\begin{minipage}{\linewidth}")
            pieces.append(r"\par\smallskip{\small\sffamily\color{gray}In ["
                          + str(cell["execution_count"]) + r"]}\nopagebreak")
            pieces.append(listing(source, "code"))
            if not keep_together and len(source.splitlines()) <= 54:
                pieces.append(r"\end{minipage}\par")
            for output in cell.get("outputs", []):
                figure = plot_output(output, build_dir)
                if figure:
                    pieces.append(figure)
                    continue
                if output["output_type"] == "stream":
                    value = "".join(output["text"])
                elif output["output_type"] in ("execute_result", "display_data"):
                    value = "".join(output["data"]["text/plain"])
                else:
                    raise ValueError("Notebook has an unexportable output")
                if not keep_together and len(value.splitlines()) <= 40:
                    pieces.append(r"\par\medskip\noindent\begin{minipage}{\linewidth}")
                    pieces.append(r"{\small\sffamily\color{gray}Output}\nopagebreak")
                    pieces.append(listing(value, "output"))
                    pieces.append(r"\end{minipage}\par\medskip")
                else:
                    pieces.append(r"{\small\sffamily\color{gray}Output}\nopagebreak")
                    pieces.append(listing(value, "output"))
            if keep_together:
                pieces.append(r"\end{minipage}\par\medskip")
    pieces.append(r"\end{document}")
    return "\n\n".join(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=shutil.which("tectonic") or shutil.which("xelatex"))
    parser.add_argument("--build-dir", type=Path,
                        default=ROOT.parents[1] / "output/ch27-solutions-build")
    args = parser.parse_args()
    if not args.engine:
        parser.error("Install Tectonic or XeLaTeX, or pass its path with --engine")
    notebook = json.loads((ROOT / "python-solutions.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code_cells]
    if counts != list(range(1, len(code_cells) + 1)):
        raise ValueError("Execute all notebook cells in order before exporting")
    if any(output["output_type"] == "error" for cell in code_cells for output in cell["outputs"]):
        raise ValueError("Notebook contains an execution error")
    headings = [int(match.group(1)) for cell in notebook["cells"]
                if (match := re.match(r"## (\d+)\.", "".join(cell["source"])))]
    if headings != list(range(1, 12)):
        raise ValueError("Expected solutions 1 through 11 in order")
    # A definition/setup cell can legitimately have no output. Each exercise
    # still needs at least one recorded result, as well as executed cells.
    results = set()
    exercise = None
    for cell in notebook["cells"]:
        match = re.match(r"## (\d+)\.", "".join(cell["source"]))
        if match:
            exercise = int(match[1])
        if exercise and cell["cell_type"] == "code" and cell.get("outputs"):
            results.add(exercise)
    if results != set(range(1, 12)):
        raise ValueError("Exercises 1 through 11 need recorded results")
    args.build_dir = args.build_dir.resolve()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for include_code in (False, True):
        stem = "python-solutions" if include_code else "written-solutions"
        tex = args.build_dir / (stem + ".tex")
        tex.write_text(document(notebook, include_code, args.build_dir.resolve()))
        if "tectonic" in Path(args.engine).name:
            command = [args.engine, "--keep-logs", "--keep-intermediates",
                       "--outdir", str(args.build_dir), str(tex)]
        else:
            command = [args.engine, "-interaction=nonstopmode", "-halt-on-error",
                       "-output-directory=" + str(args.build_dir), str(tex)]
        subprocess.run(command, check=True, cwd=args.build_dir)
        if "tectonic" not in Path(args.engine).name:
            subprocess.run(command, check=True, cwd=args.build_dir)
        log = (args.build_dir / (stem + ".log")).read_text()
        if re.search(r"Overfull|Missing character|Undefined control sequence", log):
            raise ValueError("Inspect the typesetting warnings before delivering " + stem)
        completed.append(stem)
    for stem in completed:
        shutil.copyfile(args.build_dir / (stem + ".pdf"), ROOT / (stem + ".pdf"))
        print("Exported", stem + ".pdf")


if __name__ == "__main__":
    main()
