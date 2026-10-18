"""Polymarket CS2 match prices from its public APIs (no key). Every response is cached in data/polymarket/cache/.

Gamma API lists the events (one per series: series moneyline, "Map N Winner", totals, handicaps); the CLOB API gives
each outcome token's price history. Polymarket prices are already probabilities, so there is no margin to remove.
    python3 polymarket.py events 2026-06-25 2026-10-04   # list closed CS2 events created in the window
    python3 polymarket.py fetch                          # price history for the series and map-1 markets
    python3 polymarket.py find "Team A" "Team B" [date]  # the market for that exact match and its probabilities
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


def get(url, fresh=False, **params):
    """Cached GET; fresh=True skips the cache (live prices)."""
    q = urllib.parse.urlencode(sorted(params.items()))
    f = CACHE / (hashlib.sha1(f"{url}?{q}".encode()).hexdigest() + ".json")
    if f.exists() and not fresh:
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


ALIAS = {"fazeclan": "faze", "ninjasinpyjamas": "nip", "1w": "1win"}  # Kaggle key -> Polymarket key


def key(name):
    import unicodedata
    import oos_eval as O
    k = O.key(unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode())
    return ALIAS.get(k, k)


def same_team(name, team):
    """2 = exact (name or abbreviation), 1 = close variant (prefix, small typo), 0 = different team."""
    import difflib
    k, names = key(name), {key(team["name"]), key(team.get("abbreviation") or "")}
    if k in names:
        return 2
    n = key(team["name"])
    if min(len(k), len(n)) >= 3 and (k.startswith(n) or n.startswith(k)) or difflib.SequenceMatcher(None, k, n).ratio() >= .85:
        return 1
    return 0


def find(team_a, team_b, date):
    """Every Polymarket CS2 event for these two teams starting within 36 h of `date`, best match first.

    Each item is (score, hours from `date`, event); score 4 means both names matched exactly."""
    date = pd.Timestamp(date)
    date = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
    hits = {}
    for closed in ("false", "true"):
        off = 0
        while True:
            page = get(GAMMA + "events", fresh=True, series_id=CS2_SERIES, closed=closed, limit=100, offset=off,
                       start_date_min=(date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                       start_date_max=(date + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
            for e in page:
                t = e.get("teams") or []
                if not e.get("startTime") or len(t) != 2:
                    continue
                score = max(min(same_team(team_a, x), same_team(team_b, y)) and same_team(team_a, x) + same_team(team_b, y)
                            for x, y in ((t[0], t[1]), (t[1], t[0])))
                hours = (pd.Timestamp(e["startTime"]) - date).total_seconds() / 3600
                if score and abs(hours) < 36:
                    hits[e["id"]] = (score, round(hours, 1), e)
            if len(page) < 100:
                break
            off += 100
    return sorted(hits.values(), key=lambda h: (-h[0], abs(h[1])))


def lookup(team_a, team_b, date):
    """Polymarket market for this exact match (same two teams, start within 36 h of `date`), or None.

    Before the start it returns the live price (midpoint of best bid/ask); once the match has started, the last price
    before the start, so it is never an in-play number. `spread` is best ask minus best bid: wide (> 0.2) means
    nobody is really trading and the price says little.
    """
    hits = find(team_a, team_b, date)
    if not hits:
        return None
    e, now = hits[0][2], pd.Timestamp.now(tz="UTC")
    # Same two teams twice in the window (e.g. group stage and playoff) or two close name variants: say so.
    others = [f"{h[2]['title']} ({h[1]:+.0f} h)" for h in hits[1:] if h[0] >= hits[0][0] - 1]
    start = pd.Timestamp(e["startTime"])
    out = {"event": e["title"], "url": "https://polymarket.com/event/" + e["slug"], "start": e["startTime"],
           "teams": [x["name"] for x in e["teams"]], "exact names": hits[0][0] == 4,
           "volume": round(float(e.get("volume") or 0)), "markets": {}}
    if others:
        out["other candidates"] = others
    for m in e.get("markets", []):
        q = m.get("question", "")
        name = "series" if m.get("sportsMarketType") == "moneyline" else q.rsplit(" - ", 1)[-1] if q.endswith(" Winner") else None
        if not name:
            continue
        outs, tok = json.loads(m["outcomes"]), json.loads(m["clobTokenIds"])[0]
        first_is_a = same_team(team_a, {"name": outs[0]}) >= same_team(team_b, {"name": outs[0]})
        if now < start:
            bid, ask = float(m.get("bestBid") or 0), float(m.get("bestAsk") or 1)
            p, kind = (bid + ask) / 2 if m.get("bestBid") is not None else float(json.loads(m["outcomePrices"])[0]), "live"
            spread = round(ask - bid, 3)
        else:
            p, kind, spread = closing(history(tok, e["startTime"]), start), "last price before start", None
        if p is None:
            continue
        p = p if first_is_a else 1 - p
        out["markets"][name] = {team_a: round(p, 3), team_b: round(1 - p, 3), "price": kind, "spread": spread}
    return out


def show(team_a, team_b, date):
    """Print the Polymarket line for the CLI --polymarket flag."""
    try:
        r = lookup(team_a, team_b, date)
    except Exception as e:  # no network / API change: the model prediction still stands
        print(f"polymarket: lookup failed ({e})")
        return None
    if r is None:
        print(f"polymarket: no market found for {team_a} vs {team_b} around {pd.Timestamp(date):%Y-%m-%d}")
    else:
        print("polymarket:", json.dumps(r, indent=1, ensure_ascii=False))
    return r


if __name__ == "__main__":
    if sys.argv[1] == "events":
        d = market_rows(events(sys.argv[2], sys.argv[3]))
        OUT.mkdir(parents=True, exist_ok=True)
        d.to_csv(OUT / "events.csv", index=False)
        print(len(d), "series with a moneyline;", d.map1_token.notna().sum(), "with a map-1 market")
    elif sys.argv[1] == "find":  # python3 polymarket.py find "Team Vitality" "Team Spirit" [2026-10-19T14:00]
        show(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else pd.Timestamp.now(tz="UTC"))
    elif sys.argv[1] == "fetch":
        d = load()
        for i, r in enumerate(d.itertuples()):
            history(r.series_token, r.start)
            if isinstance(r.map1_token, str):
                history(r.map1_token, r.start)
            if i % 100 == 0:
                print(i, "/", len(d))
