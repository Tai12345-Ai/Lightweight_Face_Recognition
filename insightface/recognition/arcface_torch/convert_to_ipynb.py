"""Convert colab_lightweight_fr.py (percent-format) to .ipynb notebook."""
import json, re

INPUT = r"d:\projects\Project-2\insightface\recognition\arcface_torch\colab_lightweight_fr.py"
OUTPUT = r"d:\projects\Project-2\insightface\recognition\arcface_torch\colab_lightweight_fr.ipynb"

with open(INPUT, "r", encoding="utf-8") as f:
    content = f.read()

# Split on "# %%" markers
raw_cells = re.split(r'^# %%', content, flags=re.MULTILINE)

cells = []
for i, raw in enumerate(raw_cells):
    if i == 0 and raw.strip() == "":
        continue  # skip empty preamble before first # %%

    raw = raw.rstrip("\n")

    if raw.startswith(" [markdown]"):
        # Markdown cell: lines start with "# "
        lines = raw.split("\n")[1:]  # skip " [markdown]" line itself
        md_lines = []
        for line in lines:
            if line.startswith("# "):
                md_lines.append(line[2:] + "\n")
            elif line == "#":
                md_lines.append("\n")
            else:
                md_lines.append(line + "\n")
        # Remove trailing empty lines
        while md_lines and md_lines[-1].strip() == "":
            md_lines.pop()
        if md_lines:
            md_lines[-1] = md_lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": md_lines
        })
    else:
        # Code cell
        lines = raw.split("\n")
        # Skip first line if empty (the newline after # %%)
        if lines and lines[0].strip() == "":
            lines = lines[1:]
        # Remove trailing empty lines
        while lines and lines[-1].strip() == "":
            lines.pop()
        code_lines = [l + "\n" for l in lines]
        if code_lines:
            code_lines[-1] = code_lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": code_lines,
            "outputs": [],
            "execution_count": None
        })

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Created: {OUTPUT}")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, {sum(1 for c in cells if c['cell_type']=='code')} code)")
