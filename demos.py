"""Parse CS2 demos (.dem) you downloaded into round-level and player-level tables (offline, demoparser2).

    python3 demos.py [folder]    # default data/hltv_saved/ -> data/demos/rounds.csv, data/demos/players.csv
One row per round: map, round, ct_team, t_team, winner side/team, end reason, equipment value and money per side
at freeze-end, running score. One row per player per map: kills, deaths, assists, ADR, KAST-lite (kill/assist/survive).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

ROOT = Path(__file__).parent
OUT = ROOT / "data/demos"


def parse(path, label=None):
    label = label or path.name
    p = DemoParser(str(path))
    m = p.parse_header()["map_name"].replace("de_", "").capitalize()
    ends = p.parse_event("round_end")
    ends = ends[ends.winner.isin(["CT", "T"])].sort_values("tick")
    if "round" in ends:  # a restored round (technical pause) ends twice: keep the last one
        ends = ends.drop_duplicates("round", keep="last")
    ends = ends.reset_index(drop=True)
    st = np.sort(p.parse_event("round_freeze_end").tick.values)
    # Economy at freeze-end (after buys); team names at the round-end tick (always present).
    need = sorted(set(st) | set(ends.tick - 1))
    ticks = p.parse_ticks(["team_clan_name", "team_num", "current_equip_value", "balance"], ticks=need)
    ticks = ticks[ticks.team_num.isin([2, 3])]
    rows, prev_end = [], 0
    for i, e in ends.iterrows():
        fz = st[(st > prev_end) & (st < e.tick)]  # a freeze-end inside this round; missing if recording began late
        at_end = ticks[ticks.tick == e.tick - 1]
        name = lambda n: at_end[at_end.team_num == n].team_clan_name.mode().iat[0] if (at_end.team_num == n).any() else ""
        row = {"file": label, "map": m, "round": i + 1, "ct_team": name(3), "t_team": name(2),
               "winner_side": e.winner, "winner": name(3) if e.winner == "CT" else name(2), "reason": e.reason,
               "ct_equip": np.nan, "t_equip": np.nan, "ct_money": np.nan, "t_money": np.nan,
               "end_tick": e.tick, "start_tick": fz[-1] if len(fz) else prev_end + 1}
        if len(fz):
            t = ticks[ticks.tick == fz[-1]]
            for side, n in (("ct", 3), ("t", 2)):
                row[side + "_equip"] = t[t.team_num == n].current_equip_value.sum()
                row[side + "_money"] = t[t.team_num == n].balance.sum()
        rows.append(row)
        prev_end = e.tick
    r = pd.DataFrame(rows)
    teams = sorted(set(r.ct_team) | set(r.t_team))
    for tm in teams:
        r[f"score_{tm}"] = (r.winner == tm).cumsum()
    final = sorted(r.iloc[-1][[f"score_{tm}" for tm in teams]], reverse=True)
    r["complete"] = final[0] >= 13 and final[0] - final[1] >= 2  # ponytail: regulation/OT win condition
    return r, players(p, r, label, m)


def players(p, r, fname, m):
    deaths = p.parse_event("player_death", player=["team_clan_name"])
    hurt = p.parse_event("player_hurt", player=["team_clan_name"])
    live = lambda df: df[df.tick.between(r.start_tick.min(), r.end_tick.max())].copy()
    deaths, hurt = live(deaths), live(hurt)
    # Round restores (technical pauses) kill everyone with weapon "world" on one tick; those aren't real deaths.
    mass = deaths[deaths.weapon == "world"].groupby("tick").size()
    deaths = deaths[~deaths.tick.isin(mass[mass >= 5].index)]
    rid = lambda ticks: np.searchsorted(r.end_tick.values, ticks)
    deaths["rnd"], hurt["rnd"] = rid(deaths.tick), rid(hurt.tick)
    # Damage capped at the victim's remaining HP (HLTV-style ADR; raw dmg_health includes overkill).
    hp, capped = {}, []
    for rnd, victim, dmg in zip(hurt.rnd, hurt.user_name, hurt.dmg_health):
        left = hp.get((rnd, victim), 100)
        dealt = min(dmg, left)
        hp[(rnd, victim)] = left - dealt
        capped.append(dealt)
    hurt["dmg"] = capped
    team_of = pd.concat([deaths[["attacker_name", "attacker_team_clan_name"]].set_axis(["n", "t"], axis=1),
                         deaths[["user_name", "user_team_clan_name"]].set_axis(["n", "t"], axis=1)]).dropna()
    team_of = team_of.groupby("n").t.agg(lambda s: s.mode().iat[0])
    enemy_kill = deaths.attacker_team_clan_name != deaths.user_team_clan_name
    k = deaths[enemy_kill].groupby("attacker_name").size()
    d = deaths.groupby("user_name").size()
    a = deaths[deaths.assister_name.notna()].groupby("assister_name").size() if "assister_name" in deaths else {}
    dmg = hurt[hurt.attacker_team_clan_name != hurt.user_team_clan_name].groupby("attacker_name").dmg.sum()
    n_r = len(r)
    # KAST: rounds with a Kill, Assist, Survival or being Traded (teammate kills your killer within 5 s = 320 ticks)
    kast = {n: set() for n in team_of.index}
    for rnd, g in deaths.groupby("rnd"):
        died = set(g.user_name)
        for n in kast:
            if n not in died:
                kast[n].add(rnd)
        for row in g[enemy_kill.loc[g.index]].itertuples():
            kast.setdefault(row.attacker_name, set()).add(rnd)
            if getattr(row, "assister_name", None) == getattr(row, "assister_name", None) and getattr(row, "assister_name", None):
                kast.setdefault(row.assister_name, set()).add(rnd)
            revenge = g[(g.user_name == row.attacker_name) & (g.tick > row.tick) & (g.tick <= row.tick + 320)]
            if len(revenge):
                kast.setdefault(row.user_name, set()).add(rnd)
    for rnd in range(n_r):  # rounds with no deaths at all: everyone survived
        if rnd not in set(deaths.rnd):
            for n in kast:
                kast[n].add(rnd)
    return pd.DataFrame([{"file": fname, "map": m, "player": n, "team": t, "kills": int(k.get(n, 0)),
                          "deaths": int(d.get(n, 0)), "assists": int(a.get(n, 0)) if len(a) else 0,
                          "adr": round(dmg.get(n, 0) / n_r, 1), "kast": round(100 * len(kast.get(n, ())) / n_r, 1)}
                         for n, t in team_of.items()])


def recover(f, label):
    """Some downloads arrive zero-padded with the real demo further in; copy that part out (original untouched)."""
    with open(f, "rb") as fh:
        if fh.read(7) == b"PBDEMS2":
            return f
    b = f.read_bytes()
    i = b.find(b"PBDEMS2")
    if i < 0:
        raise ValueError("no CS2 demo inside (no PBDEMS2 header)")
    out = OUT / "recovered" / label.replace("/", "__")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b[i:])
    print(f"{f.name}: demo starts at byte {i:,} (file was zero-padded), recovered {len(b) - i:,} bytes")
    return out


def main(folder):
    """Parses every .dem under `folder` (subfolders = one series each). Results are cached per demo in
    data/demos/parsed/, so adding new demos and rerunning only parses the new ones."""
    folder = Path(folder)
    cache = OUT / "parsed"
    cache.mkdir(parents=True, exist_ok=True)
    rs, ps = [], []
    files = sorted(folder.rglob("*.dem"))
    for n, f in enumerate(files, 1):
        label = str(f.relative_to(folder))
        hit = cache / (label.replace("/", "__") + ".pkl")
        try:
            if hit.exists():
                r, pl = pd.read_pickle(hit)
            else:
                r, pl = parse(recover(f, label), label)
                pd.to_pickle((r, pl), hit)
                last = r.iloc[-1]
                score = " ".join(f"{c[6:]} {last[c]}" for c in r.columns if c.startswith("score_"))
                print(f"[{n}/{len(files)}] {label}: {r['map'].iat[0]}, {len(r)} rounds, {score}", flush=True)
            rs.append(r), ps.append(pl)
        except Exception as e:
            print(f"[{n}/{len(files)}] {label}: failed ({e})", flush=True)
    R = pd.concat(rs, ignore_index=True)
    R["series"] = R.file.map(lambda x: x.split("/")[0] if "/" in x else "")
    R.drop(columns=[c for c in R.columns if c.startswith("score_")]).to_csv(OUT / "rounds.csv", index=False)
    pd.concat(ps, ignore_index=True).to_csv(OUT / "players.csv", index=False)
    print(f"{len(files)} demos, {R.file.nunique()} parsed, {R[R.complete].file.nunique()} complete maps, "
          f"{len(R)} rounds -> {OUT / 'rounds.csv'}")


def validate():
    """Compare each complete demo's final score with Liquipedia (same teams + map, within the last 120 days)."""
    from oos_eval import key
    r = pd.read_csv(OUT / "rounds.csv")
    lp = pd.read_csv(ROOT / "data/liquipedia/maps_2026.csv", parse_dates=["datetime"])
    lp = lp[lp.datetime >= lp.datetime.max() - pd.Timedelta(days=120)]
    idx = {}
    for x in lp.itertuples():
        idx.setdefault((frozenset((key(x.team1), key(x.team2))), x.map_name), []).append(
            {key(x.team1): x.score1_game, key(x.team2): x.score2_game})
    ok = bad = missing = 0
    for f, g in r[r.complete].groupby("file"):
        sc = g.winner.value_counts().to_dict()
        teams = set(g.ct_team) | set(g.t_team)
        demo = {key(t): sc.get(t, 0) for t in teams}
        cands = idx.get((frozenset(demo), g["map"].iat[0]), [])
        if not cands:
            missing += 1
        elif any(c == demo for c in cands):
            ok += 1
        else:
            bad += 1
            print("score differs:", f, demo, cands)
    print(f"vs Liquipedia: {ok} match, {bad} differ, {missing} not found")


if __name__ == "__main__":
    if sys.argv[1:] == ["validate"]:
        validate()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/hltv_saved")
