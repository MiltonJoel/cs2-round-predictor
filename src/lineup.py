"""Announced tournament lineups from Liquipedia (same API etiquette + cache as liquipedia.py).

For a match (team A, team B, date): find the tournament running that day whose participants list includes both
teams, and read each team's announced players from {{TeamParticipants}} / {{Opponent|Team|players={{Persons|...}}}}:
  - active: listed players without status=former/sub and without a staff role (coach, analyst, ...)
  - subs:   status=sub or role=sub/substitute (bench / available stand-ins)
  - former: status=former (left during the event)
  - notes:  the free-text notes (e.g. "September 27th - bobz stands in for arT ...")
Liquipedia lists the stand-in among the players once announced, so `active` is the expected five.
    python3 src/lineup.py "Legacy" "PARIVISION" 2026-10-05
"""
import re, sys

import pandas as pd

import liquipedia as L
from oos_eval import key

STAFF = re.compile(r"coach|analyst|manager|staff|strateg|director", re.I)


def _opponents(text):
    """Yield (team name, Opponent block) for every participant in a TeamParticipants section."""
    for block in L.blocks(text, "{{Opponent"):
        m = re.match(r"\{\{Opponent\|([^|\n}]+)", block)
        if m and "players=" in block:
            yield m.group(1).strip(), block


def _people(block):
    active, subs, former = [], [], []
    for p in re.findall(r"\{\{Person\|([^{}]*)\}\}", block):
        parts = [x.strip() for x in p.split("|")]
        kv = dict(x.split("=", 1) for x in parts if "=" in x)
        name = next((x for x in parts if "=" not in x), None)
        if not name:
            continue
        role, status = kv.get("role", "").split("<")[0].strip(), kv.get("status", "").strip()
        if STAFF.search(role) or kv.get("type") == "staff":
            continue
        if status == "former":
            former.append(name)
        elif status == "sub" or role.lower() in ("sub", "substitute", "stand-in", "standin"):
            subs.append(name)
        else:
            active.append(name)
    notes = []
    for n in re.findall(r"\{\{Notes\|(.*?)(?=<ref|\}\}\n|\Z)", block, re.S):
        n = re.sub(r"\{\{player\|([^|}]+)[^}]*\}\}", r"\1", n)  # {{player|bobz|flag=br}} -> bobz
        n = re.sub(r"\{\{[^{}]*\}\}|'''|\[\[|\]\]", "", n)
        notes.append(" ".join(n.split()))
    return active, subs, former, [n for n in notes if n]


def tournaments_on(date):
    """Main tournament pages whose infobox dates cover `date` (searched by month, cached)."""
    date = pd.Timestamp(date)
    months = sorted({(date - pd.Timedelta(days=d)).strftime("%Y-%m") for d in (0, 31)})
    titles = L.tournaments(months)
    texts = L.wikitext(titles)
    out = []
    for t, w in texts.items():
        s, e = L.param(w, "sdate"), L.param(w, "edate")
        try:
            if pd.Timestamp(s) <= date <= pd.Timestamp(e or s) + pd.Timedelta(days=1):
                out.append((t, w))
        except ValueError:
            continue
    return out


def announced(team_a, team_b, date):
    """{team_a: {...}, team_b: {...}, "tournament": page} or None if no running tournament lists both teams."""
    want = {key(team_a): team_a, key(team_b): team_b}
    best = None
    for title, text in tournaments_on(date):
        found = {}
        for name, block in _opponents(text):
            k = key(name)
            if k in want:
                a, s, f, n = _people(block)
                found[want[k]] = {"liquipedia_name": name, "active": a, "subs": s, "former": f, "notes": n}
        if len(found) == 2 and (best is None or len(title) > len(best[0])):  # most specific page wins
            best = (title, found)
    if best is None:
        return None
    return {"tournament": best[0], **best[1]}


if __name__ == "__main__":
    import json
    print(json.dumps(announced(*sys.argv[1:4]), indent=1, ensure_ascii=False))
