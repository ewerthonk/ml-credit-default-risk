#!/bin/python
"""Create a .py file in the same folder of the notebooks. Run the file in the terminal with
! python3 file_name.py 'name of the notebook'."""

import argparse
import re
import json
import string

parser = argparse.ArgumentParser(
    description="Print a Markdown table of contents for a Jupyter notebook."
)
parser.add_argument(
    "notebook", type=str, help="Notebook for which to create table of contents."
)
args = parser.parse_args()


if __name__ == "__main__":
    toc = []

    with open(args.notebook, "r") as f:
        cells = json.load(f)["cells"]

    for cell in cells:
        if cell["cell_type"] == "markdown":
            for line in cell["source"]:
                match = re.search("^#+ \w+", line)
                if match:
                    level = len(line) - len(line.lstrip("#"))
                    link = line.strip(" #\n").replace(" ", "-")
                    toc.append(
                        2 * (level - 1) * " "
                        + "- ["
                        + line.strip(" #\n")
                        + "](#"
                        + link
                        + ")"
                    )

    for item in toc:
        print(item)