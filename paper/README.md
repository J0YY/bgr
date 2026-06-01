# AAAI-27 Paper

`AuthorKit27` is the official AAAI-27 author kit downloaded from `https://aaai.org/authorkit27/` on 2026-06-01. The AAAI-27 conference page lists the author kit link and the main timetable at `https://aaai.org/conference/aaai/aaai-27/`.

Build the main paper from this directory:

```bash
latexmk -pdf main.tex
```

Build the reproducibility checklist separately:

```bash
pdflatex -interaction=nonstopmode ReproducibilityChecklist.tex
```

The current `main.tex` is an anonymous four-page draft with procedural and diagnostic results. It should be expanded toward the seven-page limit once learned-policy LIBERO/OpenVLA evidence is available.
