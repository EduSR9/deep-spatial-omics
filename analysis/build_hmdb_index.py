"""Make a small CSV from the full HMDB XML file (6.5 GB) with only the columns we need.

The XML is too big to search every time, so this script reads it once and keeps, for every
compound, its name, formula, exact mass, status and chemical class. hmdb_candidates.py then
searches this CSV by mass.

Run once from the project root:
    python analysis/build_hmdb_index.py

Input:  data/hmdb_metabolites.xml (downloaded from hmdb.ca)
Output: data/hmdb_compact.csv

HMDB 5.0: Wishart DS et al., Nucleic Acids Res. 2022, 50(D1), D622-D631.
https://doi.org/10.1093/nar/gkab1062
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path(__file__).resolve().parent.parent
xml_path = root / "data" / "hmdb_metabolites.xml"
out_path = root / "data" / "hmdb_compact.csv"
ns = "{http://www.hmdb.ca}"


def text(elem: ET.Element, path: str) -> str:
    """Text inside the tag at `path`, or "" if it is missing."""
    found = elem.find(path)
    if found is None or not found.text:
        return ""
    return found.text.strip()


def main() -> None:
    # 1. Columns we keep: CSV column name -> XML tag ("monisotopic" is how HMDB spells it)
    fields = {
        "accession": "accession",
        "name": "name",
        "formula": "chemical_formula",
        "monoisotopic_mass": "monisotopic_molecular_weight",
        "status": "status",
        "class": f"taxonomy/{ns}class",
        "sub_class": f"taxonomy/{ns}sub_class",
    }

    n_read = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)

        # 2. Read the XML in pieces, because the whole file does not fit in memory
        context = ET.iterparse(xml_path, events=("start", "end"))
        _, doc_root = next(context)
        for event, elem in context:
            if event != "end" or elem.tag != f"{ns}metabolite":
                continue

            # 3. Take the fields we want and write them, skipping compounds without a mass
            values = [text(elem, f"{ns}{tag}") for tag in fields.values()]
            if values[3]:
                writer.writerow(values)

            # 4. Free the memory of what was already written
            n_read += 1
            doc_root.clear()
            if n_read % 20000 == 0:
                print(f"{n_read} metabolites read", flush=True)

    print(f"Done: {n_read} metabolites -> {out_path}")


if __name__ == "__main__":
    main()
