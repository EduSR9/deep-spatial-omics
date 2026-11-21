"""HMDB candidates for the metabolites that change most between Tumor and Stroma.

Our peaks only have an m/z (mass / charge), not a name. This script tries to suggest names.

Steps:
    1. Load HMDB, the paper's annotations and the Tumor vs Stroma table from tumor_stroma.py.
    2. Keep the big-change metabolites, biggest first.
    3. For each one, turn its m/z into the mass of the neutral molecule ([M-H]- and [M+Cl]-,
       negative mode) and search HMDB for compounds with that mass.
    4. Compare the candidates with the name the paper gave to the same peak, if any.
    5. Save a table (CSV) and a readable report (HTML).

tumor_stroma.py uses search_hmdb and paper_annotation from here to name the big changes.
This script gives the detailed report, with all the candidates of each metabolite.

A candidate is only a mass match, not an identification: several compounds can share a mass.

Needs (from the project root):
    data/hmdb_compact.csv                          (python analysis/build_hmdb_index.py)
    data/annotation_reference_table1.csv           (paper table 1, optional)
    results/biology/<name>/tumor_stroma_table.csv  (python analysis/tumor_stroma.py <name>)

Run:
    python analysis/hmdb_candidates.py <name>

Output in results/biology/<name>/: hmdb_candidates.csv and hmdb_report.html.

References:
    Paper (Supporting Table 1): Godfrey TM et al. Integrating Ambient Ionization Mass Spectrometry
    Imaging and Spatial Transcriptomics on the Same Cancer Tissues to Identify RNA-Metabolite
    Correlations. Angew. Chem. Int. Ed. 2025, 64, e202502028. https://doi.org/10.1002/anie.202502028
    HMDB 5.0: Wishart DS et al., Nucleic Acids Res. 2022, 50(D1), D622-D631.
    https://doi.org/10.1093/nar/gkab1062
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent

# Settings
hmdb_path = root / "data" / "hmdb_compact.csv"
paper_path = root / "data" / "annotation_reference_table1.csv"

big_change = 1.0          # Same threshold as tumor_stroma.py
tol_ppm = 10              # Normal search window: mass error up to 10 parts per million
max_ppm = 50              # If nothing is found within tol_ppm, search up to this error...
n_wide = 3                # ...and show only the 3 closest, marked as uncertain
top_per_metabolite = 8    # Rows per metabolite in the HTML (the CSV keeps all of them)

# Ion types (adducts) in negative mode: neutral mass = m/z + shift.
# These two cover 152 of the 159 annotations in the paper.
#   [M-H]-  the molecule lost a proton, so we add its mass back
#   [M+Cl]- the molecule picked up a chloride, so we remove its mass
adducts = {"[M-H]-": 1.007276, "[M+Cl]-": -34.969402}

# Order of trust for the HMDB status: measured in humans first, predicted last
status_rank = {"quantified": 0, "detected": 1, "expected": 2, "predicted": 2}


def paper_annotation(mz: float, paper: pd.DataFrame | None) -> dict | None:
    """Annotation of the closest paper peak if it is within tol_ppm, else None."""
    if paper is None:
        return None
    ppm = np.abs(paper["Observed m/z"].to_numpy() - mz) / mz * 1e6
    closest = int(np.argmin(ppm))
    if ppm[closest] > tol_ppm:
        return None
    row = paper.iloc[closest]
    return {
        "name": row["Name"],
        "type": row["Annotation"],
        "ion": str(row["Reference Ion"]).strip(),  # Some labels in the table start with a space
        "matched_mz": float(row["Matched m/z"]),
    }


def search_hmdb(mz: float, hmdb: pd.DataFrame) -> pd.DataFrame:
    """HMDB compounds whose mass fits this m/z with any of the adducts."""
    # 1. Mass error to every HMDB compound, for each adduct
    hits = []
    for adduct, shift in adducts.items():
        neutral = mz + shift
        ppm = np.abs(hmdb["monoisotopic_mass"].to_numpy() - neutral) / neutral * 1e6
        keep = ppm <= max_ppm
        found = hmdb[keep].copy()
        found["adduct"] = adduct
        found["ion_mz"] = found["monoisotopic_mass"] - shift  # m/z this compound would give
        found["ppm"] = ppm[keep].round(2)
        hits.append(found)
    hits = pd.concat(hits, ignore_index=True)
    hits["status_rank"] = hits["status"].map(status_rank).fillna(3)

    # 2. Within tol_ppm: measured compounds first, then smallest error
    close = hits[hits["ppm"] <= tol_ppm].copy()
    if len(close):
        close["wide_match"] = False
        return close.sort_values(["status_rank", "ppm"])

    # 3. Nothing close: the n_wide closest up to max_ppm, marked as uncertain
    wide = hits.nsmallest(n_wide, "ppm").copy()
    wide["wide_match"] = True
    return wide.sort_values("ppm")


def matches_paper(cand: pd.DataFrame, ann: dict | None) -> pd.Series:
    """Same adduct and ion mass (within 3 ppm) as the paper. Compared by mass because the
    paper's formula column mixes ion and neutral formulas."""
    if ann is None:
        return pd.Series(False, index=cand.index)
    same_adduct = cand["adduct"] == ann["ion"]
    same_mass = np.abs(cand["ion_mz"] - ann["matched_mz"]) / ann["matched_mz"] * 1e6 <= 3
    return same_adduct & same_mass


def html_section(rank: int, mz: float, diff: float, r2: float, cand: pd.DataFrame,
                 ann: dict | None, agree: bool, wide: bool) -> str:
    """HTML block of one metabolite: header, paper annotation and candidate table."""
    direction = "higher in tumor" if diff > 0 else "higher in stroma"
    head = f"#{rank} | m/z {mz:.3f} | {direction} ({diff:+.2f}) | model R2 {r2:.2f}"

    if ann is None:
        paper_line = "Paper annotation: none"
    else:
        verdict = ("HMDB has a candidate with the same adduct and mass" if agree
                   else "no HMDB candidate with the same adduct and mass")
        paper_line = (f"Paper annotation: <b>{html.escape(ann['name'])}</b> "
                      f"({html.escape(ann['ion'])}, {html.escape(ann['type'])}) - {verdict}")

    # Candidates that match the paper go first so they are always shown
    shown = pd.concat([cand[cand["matches_paper"]], cand[~cand["matches_paper"]]]).head(top_per_metabolite)
    rows_html = ""
    for c in shown.itertuples():
        css = "match" if c.matches_paper else ("wide" if c.wide_match else "")
        sub_class = html.escape(str(c.sub_class)) if pd.notna(c.sub_class) else ""
        rows_html += (f"<tr class='{css}'>"
                      f"<td><a href='https://hmdb.ca/metabolites/{c.accession}'>{html.escape(str(c.name))}</a></td>"
                      f"<td>{html.escape(str(c.formula))}</td><td>{c.adduct}</td><td>{c.ppm}</td>"
                      f"<td>{html.escape(str(c.status))}</td><td>{sub_class}</td></tr>")

    if len(cand):
        table_html = ("<table><tr><th>Candidate</th><th>Formula</th><th>Adduct</th><th>ppm</th>"
                      f"<th>Status</th><th>Class</th></tr>{rows_html}</table>")
    else:
        table_html = "<p>No HMDB candidate within tolerance.</p>"
    more = (f"<p class='note'>{len(cand)} candidates in total (see CSV).</p>"
            if len(cand) > top_per_metabolite else "")
    wide_note = (f"<p class='note'>No candidate within {tol_ppm} ppm: the {len(cand)} closest up to "
                 f"{max_ppm} ppm are shown (<span class='wide'>uncertain</span>).</p>" if wide else "")
    return f"<h3>{head}</h3><p>{paper_line}</p>{wide_note}{table_html}{more}"


def write_report(path: Path, run: str, sections: list[str], n_selected: int, n_with_candidates: int,
                 n_wide_only: int, n_paper: int, n_agree: int, has_paper: bool) -> None:
    """Full HTML page: summary, how to read the tables, one block per metabolite, sources."""
    if has_paper:
        paper_summary = (f"<p>Paper comparison: {n_paper} of them were annotated in the paper; for "
                         f"{n_agree} of those, HMDB has a candidate with the same adduct and ion mass "
                         f"(<span class='match'>green rows</span>).</p>")
    else:
        paper_summary = "<p>No paper reference found: paper comparison skipped.</p>"
    summary = (f"<p>{n_selected} metabolites with a big change (|log difference| &gt;= {big_change}). "
               f"{n_with_candidates} have at least one HMDB candidate within {tol_ppm} ppm; "
               f"{n_wide_only} only have wider matches ({tol_ppm}-{max_ppm} ppm, "
               f"<span class='wide'>uncertain</span>).</p>{paper_summary}")

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>HMDB candidates - {html.escape(run)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
h3 {{ margin-top: 2rem; border-top: 1px solid #ddd; padding-top: 1rem; font-size: 1rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
th, td {{ border-bottom: 1px solid #eee; padding: 4px 6px; text-align: left; }}
th {{ background: #f5f5f5; }}
.match {{ background: #dff3e4; }}
.wide {{ background: #fff1d6; }}
.note, .warn {{ color: #666; font-size: 0.85rem; }}
</style></head><body>
<h1>HMDB candidates - {html.escape(run)}</h1>
{summary}
<p class="warn">Candidates are mass matches (negative mode, {', '.join(adducts)}, up to {tol_ppm} ppm), not identifications.
Order: measured compounds first (quantified, detected), then predicted ones, then smallest mass error.</p>
<h3>How to read the tables</h3>
<ul class="note">
<li><b>#rank</b>: position by size of the change between tumor and stroma (#1 = biggest change, either direction).</li>
<li><b>Adduct</b>: how the molecule appears in the mass spectrometer (negative mode).
[M-H]- = the molecule lost a proton; [M+Cl]- = the molecule captured a chloride ion.</li>
<li><b>ppm</b>: mass error between the measured peak and the candidate, in parts per million.
Lower is better; below 5 ppm is a close match.</li>
<li><b>Status</b>: the HMDB <code>status</code> field (labels from HMDB, descriptions are a summary,
see hmdb.ca for the exact definitions):
<ul>
<li><i>quantified</i>: measured in human samples, with a concentration.</li>
<li><i>detected</i>: observed in human samples, but not quantified.</li>
<li><i>expected</i>: expected to be present (e.g. from known pathways), but not observed.</li>
<li><i>predicted</i>: computationally predicted structure (mostly lipids), not observed.</li>
</ul></li>
<li><b>Class</b>: chemical class of the candidate, from the HMDB taxonomy.</li>
<li><span class="match">Green rows</span>: same adduct and ion mass (within 3 ppm) as the paper's annotation.
Several green rows can appear because HMDB lists isomers with the same mass.</li>
<li><span class="wide">Orange rows</span>: no candidate within {tol_ppm} ppm, so the closest ones up to {max_ppm} ppm
are shown. Treat them as uncertain.</li>
</ul>
{''.join(sections)}
<h3>Sources</h3>
<p class="note">Paper annotations: Godfrey TM et al. Integrating Ambient Ionization Mass Spectrometry Imaging and
Spatial Transcriptomics on the Same Cancer Tissues to Identify RNA-Metabolite Correlations.
<i>Angew. Chem. Int. Ed.</i> 2025, 64, e202502028.
<a href="https://doi.org/10.1002/anie.202502028">doi:10.1002/anie.202502028</a> (Supporting Table 1).<br>
HMDB 5.0: Wishart DS et al. HMDB 5.0: the Human Metabolome Database for 2022. <i>Nucleic Acids Res.</i> 2022,
50(D1), D622-D631. <a href="https://doi.org/10.1093/nar/gkab1062">doi:10.1093/nar/gkab1062</a></p>
</body></html>"""
    path.write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser(description="HMDB candidates for one tumor_stroma run")
    parser.add_argument("run", help="folder name inside results/biology/")
    args = parser.parse_args()

    out = root / "results" / "biology" / args.run
    table_path = out / "tumor_stroma_table.csv"
    for p in (hmdb_path, table_path):
        if not p.exists():
            raise SystemExit(f"Missing {p}. See the usage at the top of this file.")

    # 1. Load HMDB, the paper's annotations (optional) and the Tumor vs Stroma table
    hmdb = pd.read_csv(hmdb_path, low_memory=False)
    paper = pd.read_csv(paper_path) if paper_path.exists() else None
    if paper is None:
        print(f"No paper reference at {paper_path}, paper comparison skipped.")
    table = pd.read_csv(table_path)

    # 2. Keep the big changes, biggest first (rank 1 = biggest change, either direction)
    selected = table[table["difference_tumor_minus_stroma"].abs() >= big_change]
    selected = selected.reindex(selected["difference_tumor_minus_stroma"].abs()
                                .sort_values(ascending=False).index)
    print(f"Searching HMDB for {len(selected)} big-change metabolites ({tol_ppm} ppm, widened to "
          f"{max_ppm} ppm when needed; {', '.join(adducts)})")

    all_candidates, sections = [], []
    n_with_candidates = n_wide_only = n_paper = n_agree = 0

    for rank, (_, met) in enumerate(selected.iterrows(), start=1):
        mz, diff, r2 = met["mz"], met["difference_tumor_minus_stroma"], met["model_r2"]

        # 3. Search HMDB
        cand = search_hmdb(mz, hmdb)
        wide = bool(len(cand)) and bool(cand["wide_match"].iloc[0])
        # 4. Compare with the paper
        ann = paper_annotation(mz, paper)
        cand["matches_paper"] = matches_paper(cand, ann)
        agree = bool(cand["matches_paper"].any())

        if ann is not None:
            n_paper += 1
            n_agree += agree
        if len(cand) and not wide:
            n_with_candidates += 1
        if wide:
            n_wide_only += 1


        sections.append(html_section(rank, mz, diff, r2, cand, ann, agree, wide))
        cand.insert(0, "query_mz", mz)
        cand.insert(0, "rank", rank)
        all_candidates.append(cand)

    # 5. Save the CSV and the HTML report
    pd.concat(all_candidates, ignore_index=True).drop(columns=["status_rank"]).to_csv(
        out / "hmdb_candidates.csv", index=False)
    write_report(out / "hmdb_report.html", args.run, sections, len(selected), n_with_candidates,
                 n_wide_only, n_paper, n_agree, paper is not None)

    print(f"Paper-annotated: {n_paper} | same adduct and mass found in HMDB: {n_agree}")
    print(f"Files written to {out}")


if __name__ == "__main__":
    main()
