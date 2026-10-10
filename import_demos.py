"""After you download HLTV demos yourself: pick up the archives, parse them, refresh data.

    python3 import_demos.py [--downloads ~/Downloads] [--no-update]

1. Every .rar/.zip in --downloads whose name has "-vs-" and that contains .dem files is moved to data/hltv_saved/
   and extracted to data/hltv_saved/<archive name>/ (archives are kept; delete them yourself when you want).
2. demos.py parses the new .dem files (cached ones are not re-parsed) -> data/demos/rounds.csv, players.csv.
3. update.py pulls fresh Kaggle + Liquipedia data (skip with --no-update).
4. data/demos/demo_matches.csv (first CT/T team + frozen v1 pre-match logit per map) is rebuilt for sim.py.
"""
import argparse, shutil, subprocess
from pathlib import Path

import numpy as np

import demos, update

SAVED = demos.ROOT / "data/hltv_saved"


def listing(a):
    r = subprocess.run(["bsdtar", "-tf", str(a)], capture_output=True, text=True)
    return r.stdout.splitlines() if r.returncode == 0 else []


def extract(a, dest):
    dest.mkdir(exist_ok=True)
    r = subprocess.run(["bsdtar", "-xf", str(a), "-C", str(dest)], capture_output=True, text=True)
    if r.returncode and a.suffix == ".rar" and shutil.which("unrar"):  # libarchive misses some RAR5 variants
        r = subprocess.run(["unrar", "x", "-o+", str(a), str(dest) + "/"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or "extract failed")


def import_archives(downloads):
    new = []
    for a in sorted(Path(downloads).expanduser().glob("*-vs-*")):
        if a.suffix.lower() not in (".rar", ".zip") or not any(n.endswith(".dem") for n in listing(a)):
            continue
        dest = SAVED / a.stem
        if dest.exists():
            print(f"already imported, left in place: {a.name}")
            continue
        moved = SAVED / a.name
        shutil.move(a, moved)
        try:
            extract(moved, dest)
        except Exception as e:
            print(f"{a.name}: {e} (archive is in {SAVED})")
            continue
        n = len(list(dest.rglob("*.dem")))
        print(f"imported {a.name}: {n} demos")
        new.append(dest)
    return new


def demo_matches():
    import pandas as pd
    import sim
    r = pd.read_csv(demos.OUT / "rounds.csv")
    first = r.groupby("file").agg(series=("series", "first"), map=("map", "first"),
                                  first_ct=("ct_team", "first"), first_t=("t_team", "first")).reset_index()
    s = sim.prematch_strength(r)
    first["v1_logit_first_ct"] = first.file.map(s)
    first["v1_p_first_ct"] = 1 / (1 + np.exp(-first.v1_logit_first_ct))
    first.to_csv(demos.OUT / "demo_matches.csv", index=False)
    print(f"demo_matches.csv: {len(first)} maps, {first.v1_logit_first_ct.notna().sum()} with a pre-match logit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", default="~/Downloads")
    ap.add_argument("--no-update", action="store_true")
    a = ap.parse_args()
    new = import_archives(a.downloads)
    print(f"{len(new)} new archives")
    demos.main(SAVED)
    if not a.no_update:
        update.run()
    demo_matches()
