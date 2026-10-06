"""CS2 player transfers from Liquipedia ("Player Transfers/YYYY/Month" pages) -> dated player moves.

Each row: date, player (lower-case name), from_team, to_team (Liquipedia team names, lower-case),
active_to (True when the player joins as an active/stand-in player, False for inactive/benched/sub).
Non-player roles (coach, analyst, manager, ...) are dropped.
    python3 rosters.py   # fetch (cached) + write data/liquipedia/transfers.csv
"""
import html, re
from pathlib import Path

import pandas as pd

import liquipedia as L

OUT = Path(__file__).parent / "data/liquipedia/transfers.csv"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
STAFF = re.compile(r"coach|analyst|manager|director|ceo|owner|strateg|psycholog|performance|content|caster", re.I)
INACTIVE = re.compile(r"inactive|bench|substitute|sub\b|retired", re.I)


def pages(today=None):
    today = pd.Timestamp.today() if today is None else today
    return [f"Player Transfers/{y}/{MONTHS[m - 1]}" for y in range(2024, today.year + 1) for m in range(1, 13)
            if (y, m) <= (today.year, today.month)]


def team_names(ids):
    names = {}
    ids = sorted(i for i in ids if i)
    for i in range(0, len(ids), 100):
        r = L.api(action="expandtemplates", prop="wikitext",
                  text="\n@@\n".join("{{TeamBracket|%s}}" % x for x in ids[i:i + 100]))
        for x, n in zip(ids[i:i + 100], r["expandtemplates"]["wikitext"].split("\n@@\n")):
            m = re.search(r'data-highlightingclass="([^"]+)"', n)
            names[x] = html.unescape(m.group(1)).lower() if m else x
    return names


def collect(titles=None):
    rows = []
    for page, text in L.wikitext(titles or pages()).items():
        for b in L.blocks(text, "{{Transfer Row"):
            p = lambda k: L.param(b, k)
            role1, role2 = p("role1"), p("role2")
            if STAFF.search(role1 + " " + role2) or not p("date"):
                continue
            for k in ("name", "name2", "name3", "name4", "name5"):
                if p(k):
                    rows.append({"date": p("date"), "player": p(k).lower(), "from_team": p("team1").lower(),
                                 "to_team": p("team2").lower(), "role1": role1, "role2": role2,
                                 "active_to": bool(p("team2")) and not INACTIVE.search(role2)})
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d.date, errors="coerce")
    d = d.dropna(subset=["date"])
    names = team_names(set(d.from_team) | set(d.to_team))
    for c in ("from_team", "to_team"):
        d[c] = d[c].map(lambda x: names.get(x, x) if x else "")
    return d.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    d = collect()
    d.to_csv(OUT, index=False)
    print(len(d), "player moves", d.date.min().date(), "->", d.date.max().date(), "| active joins",
          int(d.active_to.sum()), "| leaves", int((d.from_team != "").sum()))
