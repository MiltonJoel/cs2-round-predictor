"""Pinnacle CS2 odds from OddsPapi (free plan: 250 requests/month, 5 s cooldown on historical-odds).

Key lives in .oddspapi_key (gitignored). Every response is cached in data/odds/cache/, so reruns are free.
    python3 odds.py fixtures 2026-07-01 2026-10-03   # list CS2 fixtures (1 request per window)
    python3 odds.py fetch 200                        # historical odds for up to N matched fixtures
"""
import gzip, hashlib, json, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
CACHE = ROOT / "data/odds/cache"
API = "https://api.oddspapi.io/v4/"
CS2 = 17
WINNER, MAP1 = "171", "1747"  # series winner, first map winner (outcome ids: 171/172 and 1747/1748)
_last = [0.0]


def get(endpoint, **params):
    """Cached GET. The cache key never includes the API key."""
    q = urllib.parse.urlencode(sorted(params.items()))
    f = CACHE / (hashlib.sha1(f"{endpoint}?{q}".encode()).hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text())
    time.sleep(max(0, 5.5 - (time.time() - _last[0])))
    key = (ROOT / ".oddspapi_key").read_text().strip()
    url = API + endpoint + "?" + urllib.parse.urlencode({**params, "apiKey": key})
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "CounterPredictor/0.1 (research)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise RuntimeError(f"{endpoint} HTTP {e.code}: {e.read()[:200]!r}") from None
        body = b"null"  # cache 'no odds' too, so we never pay for it twice
    _last[0] = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_bytes(body)
    return json.loads(body)


def fixtures(start, end):
    return get("fixtures", sportId=CS2, **{"from": start, "to": end}) or []


def history(fixture_id):
    return get("historical-odds", fixtureId=fixture_id, bookmakers="pinnacle")


def closing(hist, market, before):
    """Pinnacle no-vig P(participant1) from the last prices created before `before` (UTC ISO string)."""
    try:
        outs = hist["bookmakers"]["pinnacle"]["markets"][market]["outcomes"]
    except (TypeError, KeyError):
        return None
    o1, o2 = sorted(outs, key=int)
    # Both prices from the same snapshot (the last one before `before` where both outcomes were quoted).
    p1 = {e["createdAt"]: e["price"] for e in outs[o1]["players"]["0"] if e["createdAt"] < before and (e["price"] or 0) > 1}
    p2 = {e["createdAt"]: e["price"] for e in outs[o2]["players"]["0"] if e["createdAt"] < before and (e["price"] or 0) > 1}
    common = sorted(set(p1) & set(p2))
    if not common:
        return None
    last = [p1[common[-1]], p2[common[-1]]]
    i1, i2 = 1 / last[0], 1 / last[1]
    return i1 / (i1 + i2)


if __name__ == "__main__":
    if sys.argv[1] == "fixtures":
        f = fixtures(sys.argv[2], sys.argv[3])
        print(len(f), "fixtures", sum(x.get("statusId") is not None for x in f), "with status")
