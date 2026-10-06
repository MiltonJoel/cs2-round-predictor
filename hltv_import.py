"""Offline reader for HLTV match pages you saved yourself (browser: Ctrl+S, "Webpage, complete" or "HTML only").

No network access: it only parses .html files in data/hltv_saved/ and writes data/hltv_saved/maps.csv
(one row per map, same columns the models use: datetime, event, team1, team2, map_name, score1_game,
score2_game, bestOf, map_no, team{1,2}_player{1..5}[_kills/_deaths/_adr/_kast], plus veto info when present).
    python3 hltv_import.py
"""
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

DIR = Path(__file__).parent / "data/hltv_saved"
MAPS = {"dust2": "Dust2", "de_dust2": "Dust2"}


def txt(el):
    return el.get_text(" ", strip=True) if el else ""


def num(s):
    m = re.search(r"-?\d+(\.\d+)?", s or "")
    return float(m.group()) if m else None


def parse_veto(soup, t1, t2):
    """'1. Vitality removed Ancient' / '3. Spirit picked Mirage' / '7. Dust2 was left over' -> {map: picker}."""
    picks = {}
    for line in soup.select(".veto-box div, .veto-box p"):
        s = txt(line)
        m = re.match(r"\d+\.\s*(.+?)\s+picked\s+(\w[\w ]*)$", s)
        if m:
            picks[m.group(2).strip()] = m.group(1).strip()
        m = re.match(r"\d+\.\s*(\w[\w ]*?)\s+was left over", s)
        if m:
            picks[m.group(1).strip()] = "decider"
    return picks


def player_stats(table):
    """HLTV stats table -> list of (nick, kills, deaths, adr, kast) in table order."""
    out = []
    for tr in table.select("tr"):
        nick = txt(tr.select_one(".player-nick")) or txt(tr.select_one(".players"))
        kd = txt(tr.select_one(".kd"))
        if not nick or "-" not in kd:
            continue
        k, d = (num(x) for x in kd.split("-")[:2])
        out.append((nick, k, d, num(txt(tr.select_one(".adr"))), num(txt(tr.select_one(".kast")))))
    return out[:5]


def parse(path):
    soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
    names = [txt(x) for x in soup.select(".teamsBox .teamName")][:2]
    if len(names) < 2:
        return []
    t1, t2 = names
    unix = soup.select_one(".timeAndEvent .date[data-unix], .timeAndEvent .time[data-unix]")
    when = datetime.fromtimestamp(int(unix["data-unix"]) / 1000, timezone.utc).replace(tzinfo=None) if unix else None
    event = txt(soup.select_one(".timeAndEvent .event a, .timeAndEvent .event"))
    fmt = txt(soup.select_one(".preformatted-text, .veto-box .padding"))
    bo = int(m.group(1)) if (m := re.search(r"[Bb]est of (\d)", fmt)) else None
    picks = parse_veto(soup, t1, t2)
    # Per-map stats tables: div.stats-content with id "<mapstatsid>-content"; the first one ("all-content") is totals.
    stat_divs = [d for d in soup.select("div.stats-content") if d.get("id") != "all-content"]
    rows = []
    for k, mh in enumerate(soup.select(".mapholder"), 1):
        mp = txt(mh.select_one(".mapname"))
        scores = [num(txt(x)) for x in mh.select(".results-team-score")]
        if not mp or len(scores) < 2 or None in scores[:2] or mp.lower() in ("tba", "default"):
            continue
        row = {"source_file": path.name, "datetime": when, "event": event, "team1": t1, "team2": t2,
               "map_name": MAPS.get(mp.lower(), mp), "score1_game": scores[0], "score2_game": scores[1],
               "bestOf": bo, "map_no": k, "picked_by": picks.get(mp, "")}
        if k - 1 < len(stat_divs):
            tables = stat_divs[k - 1].select("table.totalstats")
            for side, tb in zip((1, 2), tables[:2]):
                for i, (nick, kills, deaths, adr, kast) in enumerate(player_stats(tb), 1):
                    row.update({f"team{side}_player{i}": nick, f"team{side}_player{i}_kills": kills,
                                f"team{side}_player{i}_deaths": deaths, f"team{side}_player{i}_adr": adr,
                                f"team{side}_player{i}_kast": kast})
        rows.append(row)
    return rows


def main():
    files = sorted(DIR.glob("*.htm*"))
    rows = [r for f in files for r in parse(f)]
    if not rows:
        print(f"no match pages parsed from {len(files)} files in {DIR}")
        return
    d = pd.DataFrame(rows).drop_duplicates(["datetime", "team1", "team2", "map_no"])
    kaggle = DIR.parent / "kaggle_v5/cs2_all_tiers_games.csv"
    if kaggle.exists():  # map HLTV names onto the Kaggle team names the models use ("Vitality" -> "Team Vitality")
        from oos_eval import key
        k = pd.read_csv(kaggle, usecols=["team1", "team2"])
        known = {}
        for n in pd.concat([k.team1, k.team2]):
            known.setdefault(key(n), n)
        for c in ("team1", "team2"):
            d[c] = d[c].map(lambda x: known.get(key(x), x))
        d["picked_by"] = d.picked_by.map(lambda x: known.get(key(x), x) if x not in ("", "decider") else x)
    d.to_csv(DIR / "maps.csv", index=False)
    print(f"{len(files)} files -> {len(d)} maps ({d.datetime.min()} .. {d.datetime.max()}), "
          f"{d.team1_player1.notna().mean() if 'team1_player1' in d else 0:.0%} with player stats, "
          f"{(d.picked_by != '').mean():.0%} with veto -> {DIR / 'maps.csv'}")


if __name__ == "__main__":
    main()
