"""Polymarket CS2 match prices from its public APIs (no key). Every response is cached in data/polymarket/cache/.

Gamma API lists the events (one per series: series moneyline, "Map N Winner", totals, handicaps); the CLOB API gives
each outcome token's price history. Polymarket prices are already probabilities, so there is no margin to remove.
    python3 polymarket.py events 2026-06-25 2026-10-04   # list closed CS2 events created in the window
    python3 polymarket.py fetch                          # price history for the series and map-1 markets
"""
import hashlib, json, sys, time, urllib.parse, urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "data/polymarket"
CACHE = OUT / "cache"
GAMMA, CLOB = "https://gamma-api.polymarket.com/", "https://clob.polymarket.com/"
CS2_SERIES = 10310
_last = [0.0]


def get(url, **params):
    q = urllib.parse.urlencode(sorted(params.items()))
    f = CACHE / (hashlib.sha1(f"{url}?{q}".encode()).hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text())
    time.sleep(max(0, 0.3 - (time.time() - _last[0])))  # well under the published rate limits
    req = urllib.request.Request(url + "?" + q, headers={"User-Agent": "CounterPredictor/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    _last[0] = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_bytes(body)
    return json.loads(body)


def events(start, end):
    """Closed CS2 events created in [start, end), fetched week by week (the API caps offset at 3000)."""
    out, days = [], pd.date_range(start, end, freq="7D").append(pd.DatetimeIndex([pd.Timestamp(end)]))
    for a, b in zip(days[:-1], days[1:]):
        off = 0
        while True:
            page = get(GAMMA + "events", series_id=CS2_SERIES, closed="true", start_date_min=a.strftime("%Y-%m-%d"),
                       start_date_max=b.strftime("%Y-%m-%d"), limit=100, offset=off, order="startDate", ascending="true")
            out += page
            if len(page) < 100:
                break
            off += 100
    return list({e["id"]: e for e in out}.values())


def market_rows(evs):
    """One row per series: teams, start time, token ids of team1 for the series and map-1 markets."""
    rows = []
    for e in evs:
        if not e.get("startTime") or len(e.get("teams") or []) != 2:
            continue
        r = {"event_id": e["id"], "slug": e["slug"], "title": e["title"], "start": e["startTime"],
             "league": (e.get("eventMetadata") or {}).get("league"), "tier": (e.get("eventMetadata") or {}).get("leagueTier"),
             "volume": e.get("volume")}
        for m in e.get("markets", []):
            kind = m.get("sportsMarketType")
            name = "series" if kind == "moneyline" else "map1" if m.get("question", "").endswith("Map 1 Winner") else None
            if not name:
                continue
            outs, toks = json.loads(m["outcomes"]), json.loads(m["clobTokenIds"])
            r["team1"], r["team2"] = outs
            r[name + "_token"] = toks[0]
            r[name + "_volume"] = float(m.get("volume") or 0)
            r[name + "_result"] = json.loads(m["outcomePrices"])[0]
        if "series_token" in r:
            rows.append(r)
    return pd.DataFrame(rows)


def load():
    return pd.read_csv(OUT / "events.csv", dtype={"series_token": str, "map1_token": str, "event_id": str})


def history(token, start):
    """Price points of `token` from 10 days before `start` up to 6 hours after (enough to see the pre-match close)."""
    t = int(pd.Timestamp(start).timestamp())
    return get(CLOB + "prices-history", market=token, startTs=t - 10 * 86400, endTs=t + 6 * 3600, fidelity=10).get("history", [])


def closing(hist, before):
    """Last traded/mid price strictly before `before` (UTC timestamp), or None."""
    t = pd.Timestamp(before).timestamp()
    pts = [h["p"] for h in hist if h["t"] < t]
    return pts[-1] if pts else None


if __name__ == "__main__":
    if sys.argv[1] == "events":
        d = market_rows(events(sys.argv[2], sys.argv[3]))
        OUT.mkdir(parents=True, exist_ok=True)
        d.to_csv(OUT / "events.csv", index=False)
        print(len(d), "series with a moneyline;", d.map1_token.notna().sum(), "with a map-1 market")
    elif sys.argv[1] == "fetch":
        d = load()
        for i, r in enumerate(d.itertuples()):
            history(r.series_token, r.start)
            if isinstance(r.map1_token, str):
                history(r.map1_token, r.start)
            if i % 100 == 0:
                print(i, "/", len(d))
