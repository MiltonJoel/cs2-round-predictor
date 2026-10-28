"""Fetch CS2 map results from Liquipedia's MediaWiki API (independent source for out-of-sample checks).

Follows https://liquipedia.net/api-terms-of-use: descriptive User-Agent, gzip, >=2 s between requests,
no action=parse. Every response is cached under data/liquipedia/ so reruns cost zero requests.
"""
import gzip, hashlib, json, time, urllib.parse, urllib.request
from pathlib import Path

API = "https://liquipedia.net/counterstrike/api.php"
UA = "CounterPredictor/0.1 (research project; https://github.com/MiltonJoel)"
CACHE = Path(__file__).resolve().parent.parent / "data/liquipedia/cache"
_last = [0.0]
FRESH = False  # True: ignore cached responses (still rewrites the cache) -- used by update.py for recent months


def api(**params):
    params = {"format": "json", "formatversion": 2, **params}
    url = API + "?" + urllib.parse.urlencode(sorted(params.items()))
    f = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".json")
    if f.exists() and not FRESH:
        return json.loads(f.read_text())
    time.sleep(max(0, 2.5 - (time.time() - _last[0])))
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    _last[0] = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_bytes(body)
    return json.loads(body)


def wikitext(titles):
    """{title: wikitext} for up to 50 titles in one request (follows redirects)."""
    out = {}
    for i in range(0, len(titles), 50):
        r = api(action="query", prop="revisions", rvprop="content", rvslots="main", redirects=1,
                titles="|".join(titles[i:i + 50]))
        for p in r["query"]["pages"]:
            if "revisions" in p:
                out[p["title"]] = p["revisions"][0]["slots"]["main"]["content"]
    return out


import html, re
from datetime import datetime

import pandas as pd

TIER = {"S-Tier": 1, "A-Tier": 1, "B-Tier": 2, "C-Tier": 3, "1": 1, "2": 1, "3": 2, "4": 3}  # ponytail: rough map onto Kaggle's 3 tiers


def map_name(m):
    m = m.replace("de_", "").replace(" ", "").replace("II", "2").capitalize()
    return m


def tournaments(months):
    titles = set()
    for m in months:
        for field in ("sdate", "edate"):
            off = 0
            while off is not None:
                r = api(action="query", list="search", srsearch=f'insource:"{field}={m}"', srlimit=500,
                        srnamespace=0, sroffset=off)
                titles |= {p["title"] for p in r["query"]["search"]}
                off = r.get("continue", {}).get("sroffset")
    return sorted(titles)


def blocks(text, start):
    """Balanced {{...}} blocks beginning with `start` as a whole template name ({{Match, not {{Matchlist)."""
    pat = re.compile(re.escape(start) + r"\s*[|\n]")
    m = pat.search(text)
    while m:
        i = m.start()
        depth, j = 0, i
        while j < len(text) - 1:
            if text.startswith("{{", j):
                depth, j = depth + 1, j + 2
            elif text.startswith("}}", j):
                depth, j = depth - 1, j + 2
                if depth == 0:
                    break
            else:
                j += 1
        yield text[i:j]
        m = pat.search(text, j)


def param(block, name):
    m = re.search(r"\|\s*" + name + r"\s*=([^|}]*)", block)
    return m.group(1).strip() if m else ""


def parse_date(s):
    m = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})(?:\s*-\s*(\d{1,2}:\d{2}))?", s)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + " " + (m.group(2) or "00:00"), "%B %d, %Y %H:%M")
    except ValueError:
        return None


def matches(page, text, info):
    for mb in blocks(text, "{{Match"):
        opp = re.findall(r"opponent[12]=\{\{TeamOpponent\|([^|}]+)", mb)
        when = parse_date(param(mb, "date"))
        if len(opp) != 2 or when is None or "walkover" in mb:
            continue
        maps = []
        for mp in re.findall(r"\{\{Map\|(.*?)\}\}", mb, re.S):
            p = lambda k: param("|" + mp, k)
            if p("finished") in ("skip", "") or not p("map"):
                continue
            if p("score1"):
                s1, s2 = p("score1"), p("score2")
            else:
                num = lambda ks: sum(int(v) for k in ks for v in re.findall(r"\|\s*" + k + r"\s*=\s*(\d+)", "|" + mp))
                s1, s2 = num([r"(?:o\d+)?t1t", r"(?:o\d+)?t1ct"]), num([r"(?:o\d+)?t2t", r"(?:o\d+)?t2ct"])
            try:
                s1, s2 = int(s1), int(s2)
            except ValueError:
                continue
            maps.append((map_name(p("map")), s1, s2))
        for k, (m, s1, s2) in enumerate(maps):
            yield {"page": page, "event": info["name"], "tier_lp": info["tier"], "datetime": when,
                   "team1_lp": opp[0].strip().lower(), "team2_lp": opp[1].strip().lower(), "map_name": m,
                   "score1_game": s1, "score2_game": s2, "map_no": k + 1, "bestOf": len(re.findall(r"\|map\d+=", mb)),
                   "hltv_match": param(mb, "hltv")}


def collect(months):
    tours = tournaments(months)
    texts = wikitext(tours)
    sub = set()
    for t, w in texts.items():
        sub |= {s.strip() for s in re.findall(r"\|link\d+\s*=([^|}\n]+)", w) if s.strip().startswith(t + "/")}
        sub |= {s.strip() for s in re.findall(r"\{\{:([^|}]+)\}\}", w)}
    texts.update(wikitext(sorted(sub - set(texts))))
    rows = []
    for page, w in texts.items():
        root = next((t for t in sorted(tours, key=len, reverse=True) if page == t or page.startswith(t + "/")), page)
        ib = texts.get(root, w)
        info = {"name": html.unescape(param(ib, "name") or root), "tier": param(ib, "liquipediatier")}
        rows += list(matches(page, w, info))
    d = pd.DataFrame(rows).drop_duplicates(["datetime", "team1_lp", "team2_lp", "map_no"])
    ids = sorted(set(d.team1_lp) | set(d.team2_lp))
    names = {}
    for i in range(0, len(ids), 100):  # {{TeamBracket|id}} carries the canonical team name in data-highlightingclass
        r = api(action="expandtemplates", prop="wikitext",
                text="\n@@\n".join("{{TeamBracket|%s}}" % x for x in ids[i:i + 100]))
        for x, n in zip(ids[i:i + 100], r["expandtemplates"]["wikitext"].split("\n@@\n")):
            m = re.search(r'data-highlightingclass="([^"]+)"', n)
            if m:
                names[x] = html.unescape(m.group(1)).lower()
    d["team1"] = d.team1_lp.map(lambda x: names.get(x, x))
    d["team2"] = d.team2_lp.map(lambda x: names.get(x, x))
    d["tier_lp"] = d.tier_lp.replace({"1": "S-Tier", "2": "A-Tier", "3": "B-Tier", "4": "C-Tier"})
    d["tier"] = d.tier_lp.map(TIER)
    return d.sort_values(["datetime", "team1", "map_no"]).reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data/liquipedia/maps_2026.csv"
    d = collect([f"2026-{m:02d}" for m in range(1, 11)])
    d.to_csv(out, index=False)
    print(len(d), "maps", d.datetime.min(), d.datetime.max(), "->", out)
