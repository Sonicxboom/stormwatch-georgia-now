"""
Georgia Now, live: every hour, the newest two of NOAA's real-time weather maps of Georgia (RTMA, 2.5 km) go into the
Georgia Now network, which steps them forward 18 hours. The forecast at each of your places goes to your StormWatch
server (address and key come from this repo's secrets and never appear in the log).

  python run.py          # the hourly run
  python run.py --test   # same, but prints a sample for Stone Mountain and sends nothing
  python run.py --backup # GitHub's run: skips when the Modal run already sent this hour

The network was trained on URMA (the same maps, finalized hours later); RTMA is the version that's out within the hour.

StormWatchNow: when SW_NOW_MODEL points to a StormWatchNow checkpoint (the Modal run passes the promoted one from Modal
storage), it is used instead of Georgia Now. It reads the last 24 hours of maps (now, 1, 2, 3, 6, 12 and 24 hours ago)
and forecasts 1, 2, 3, 6, 9, 12, 15 and 18 hours ahead in one pass; the hours in between are filled in on a straight
line. If any of those older maps is missing, the run falls back to Georgia Now.
"""
import os, sys, json, math, time, datetime as dt, urllib.request, urllib.error
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL, STATIC = os.path.join(HERE, 'model', 'georgia_now.pt'), os.path.join(HERE, 'model', 'static.npz')
BUCKET = 'https://noaa-rtma-pds.s3.amazonaws.com'
UA = {'User-Agent': 'StormWatch personal weather app (Georgia Now)'}
FIELDS = [('TMP', '2 m above ground'), ('DPT', '2 m above ground'), ('UGRD', '10 m above ground'),
          ('VGRD', '10 m above ground'), ('GUST', '10 m above ground'), ('TCDC', 'entire atmosphere')]
NF, SIZE, ROLL = len(FIELDS), 256, 18
TEST = '--test' in sys.argv
BACKUP = '--backup' in sys.argv   # GitHub's run: send only if the main (Modal) run hasn't in the last hour

def clean(v): return str(v or '').strip().strip('"\'` ,;').rstrip('/')
API, KEY = clean(os.environ.get('STORMWATCH_API')), clean(os.environ.get('STORMWATCH_KEY'))
if API and not API.lower().startswith('http'): API = 'https://' + API

def get(url, rng=None, tries=4, headers=None):
    h = dict(UA, **(headers or {}))
    if rng: h['Range'] = f'bytes={rng[0]}-{rng[1]}'
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=90) as r: return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404): return None
            if k == tries - 1: raise
        except Exception:
            if k == tries - 1: raise
        time.sleep(4 * (k + 1))

def hour_url(t): return f'{BUCKET}/rtma2p5.{t:%Y%m%d}/rtma2p5.t{t:%H}z.2dvaranl_ndfd.grb2_wexp'

def byte_ranges(idx_txt):
    lines = [l.split(':') for l in idx_txt.strip().splitlines()]; out = {}
    for i, c in enumerate(lines):
        for w in FIELDS:
            if w not in out and c[3] == w[0] and c[4].startswith(w[1]):
                out[w] = (int(c[1]), int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else '')
    return out

_crop = {}
def decode(buf, st):
    import pygrib
    g = pygrib.fromstring(buf)
    if 'box' not in _crop:   # the same 256 x 256 window Georgia Now was trained on, found by matching its lat/lons
        la, lo = g.latlons(); tla, tlo = st['lat'], st['lon']
        d = (la - tla[0, 0]) ** 2 + (lo - tlo[0, 0]) ** 2; r0, c0 = (int(v) for v in np.unravel_index(np.argmin(d), d.shape))
        ok = np.abs(la[r0:r0 + SIZE, c0:c0 + SIZE] - tla).max() < 0.01 and np.abs(lo[r0:r0 + SIZE, c0:c0 + SIZE] - tlo).max() < 0.01
        if not ok: raise SystemExit('RTMA grid does not line up with Georgia Now\'s map')
        _crop['box'] = (r0, c0)
    r0, c0 = _crop['box']; v = g.values
    v = np.ma.filled(v.astype(np.float32), np.nan) if np.ma.isMaskedArray(v) else v.astype(np.float32)
    return v[r0:r0 + SIZE, c0:c0 + SIZE]

def fetch_hour(t, st):
    idx = get(hour_url(t) + '.idx')
    if not idx: return None
    R = byte_ranges(idx.decode())
    if len(R) < NF: return None
    out = []
    for f in FIELDS:
        b = get(hour_url(t), R[f])
        if not b: return None
        out.append(decode(b, st))
    return np.stack(out)

def build_net(cin, cout, w=32, levels=4):   # the same network as deeplearn/georgia_now.py
    import torch, torch.nn as nn
    def blk(a, b): return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(), nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.L = levels; s.pool = nn.AvgPool2d(2)
            for k in range(1, levels + 1): setattr(s, f'e{k}', blk(cin if k == 1 else w * 2 ** (k - 2), w * 2 ** (k - 1)))
            for k in range(levels - 1, 0, -1):
                setattr(s, f'u{k}', nn.ConvTranspose2d(w * 2 ** k, w * 2 ** (k - 1), 2, 2)); setattr(s, f'd{k}', blk(w * 2 ** k, w * 2 ** (k - 1)))
            s.out = nn.Conv2d(w, cout, 1)
        def forward(s, x):
            E = [s.e1(x)]
            for k in range(2, s.L + 1): E.append(getattr(s, f'e{k}')(s.pool(E[-1])))
            d = E[-1]
            for k in range(s.L - 1, 0, -1): d = getattr(s, f'd{k}')(torch.cat([getattr(s, f'u{k}')(d), E[k - 1]], 1))
            return s.out(d)
    return Net()

def time_ch(t):
    h, y = 2 * math.pi * t.hour / 24, 2 * math.pi * t.timetuple().tm_yday / 365.25
    return [math.sin(h), math.cos(h), math.sin(y), math.cos(y)]

def now_forecast(path, t0, cur, st):
    """StormWatchNow: (18, NF, H, W) in real units from the last 24 hours of maps, or None (then Georgia Now runs)."""
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('plus'): print('(StormWatchNow+ needs the HRRR maps; not supported in the hourly run yet)'); return None
    SC = np.array(ck['scale'], np.float32); CE = tuple(ck['center']); LEADS, HIST = ck['leads'], ck['hist']
    maps = []
    for h in HIST:
        m = cur if h == 0 else fetch_hour(t0 - dt.timedelta(hours=h), st)
        if m is None: print(f'(the map from {h} h ago is missing; using Georgia Now this hour)'); return None
        maps.append(m)
    net = build_net(ck['cin'], len(LEADS) * ck['nf'], w=ck['width'], levels=ck['levels']); net.load_state_dict(ck['state']); net.eval()
    sc = lambda a: (a - SC[:, 0, None, None]) / SC[:, 1, None, None]
    M = [torch.nan_to_num(torch.tensor(sc(m)))[None] for m in maps]
    S = torch.tensor(np.stack([st['hgt'].astype(np.float32) / 1000.0, (st['lat'] - CE[0]) / 3.0, (st['lon'] - CE[1]) / 3.0]).astype(np.float32))
    tc = torch.tensor(time_ch(t0))[None, :, None, None].expand(-1, -1, SIZE, SIZE)
    with torch.no_grad():
        d = net(torch.cat(M + [S[None], tc], 1)).view(len(LEADS), NF, SIZE, SIZE)
    P = (M[0][0][None] + d).numpy()                                           # (leads, NF, H, W), scaled
    P = P * SC[None, :, 1, None, None] + SC[None, :, 0, None, None]            # real units
    xs, out = [0] + list(LEADS), []
    for k in range(1, ROLL + 1):                                               # straight line between the forecast hours
        j = next(i for i in range(1, len(xs)) if xs[i] >= k); a, b = xs[j - 1], xs[j]; w = (k - a) / (b - a)
        lo = cur if a == 0 else P[j - 2]; hi = P[j - 1]
        out.append(hi if w == 1 else lo * (1 - w) + hi * w)
    return np.stack(out).astype(np.float32)

def main():
    import torch
    torch.set_num_threads(os.cpu_count() or 2)
    ck = torch.load(MODEL, map_location='cpu', weights_only=False); st = np.load(STATIC)
    SCALE = np.array(ck['scale'], np.float32); CENTER = tuple(ck['center'])
    net = build_net(ck['cin'], ck['nf']); net.load_state_dict(ck['state']); net.eval()
    S = torch.tensor(np.stack([st['hgt'].astype(np.float32) / 1000.0, (st['lat'] - CENTER[0]) / 3.0, (st['lon'] - CENTER[1]) / 3.0]).astype(np.float32))
    NOW = os.environ.get('SW_NOW_MODEL', '')
    # the newest hour whose map (and the one before it) is out
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    t0 = cur = prev = None
    for back in range(0, 5):
        t = now - dt.timedelta(hours=back)
        if get(hour_url(t) + '.idx') is None: continue
        cur = fetch_hour(t, st); prev = fetch_hour(t - dt.timedelta(hours=1), st) if cur is not None else None
        if cur is not None and prev is not None: t0 = t; break
    if t0 is None: print('No fresh RTMA maps yet; trying again next hour.'); return 0
    print(f'Map: {t0:%Y-%m-%d %H}z ({(now - t0).total_seconds() / 60:.0f} min old at the top of the hour)', flush=True)
    tt, F, model = time.time(), None, 'Georgia Now'
    if NOW and os.path.exists(NOW):
        try: F = now_forecast(NOW, t0, cur, st)
        except Exception as e: print('(StormWatchNow failed:', e, '; using Georgia Now)')
        if F is not None: model = 'StormWatchNow'
    if F is None:
        sc = lambda a: (a - SCALE[:, 0, None, None]) / SCALE[:, 1, None, None]
        P, C = torch.tensor(sc(prev))[None], torch.tensor(sc(cur))[None]
        out = []
        with torch.no_grad():
            for k in range(1, ROLL + 1):
                tc = torch.tensor(time_ch(t0 + dt.timedelta(hours=k - 1)))[None, :, None, None].expand(-1, -1, SIZE, SIZE)
                x = torch.cat([torch.nan_to_num(C), torch.nan_to_num(C - P), S[None], tc], 1)
                N = torch.nan_to_num(C) + net(x); out.append(N[0].numpy()); P, C = C, N
        F = np.stack(out) * SCALE[None, :, 1, None, None] + SCALE[None, :, 0, None, None]   # (18, NF, H, W) real units
    print(f'{model}: 18 hours forecast in {time.time() - tt:.0f} s', flush=True)
    def at(la, lo):   # 3x3 mean around a spot -> leads, or None if it's off the map
        d = (st['lat'] - la) ** 2 + (((st['lon'] - lo + 180) % 360 - 180) * math.cos(math.radians(la))) ** 2
        r, c = (int(v) for v in np.unravel_index(np.argmin(d), d.shape))
        if d[r, c] > 0.03 ** 2 or not (8 <= r < SIZE - 8 and 8 <= c < SIZE - 8): return None
        m = np.nanmean(F[:, :, r - 1:r + 2, c - 1:c + 2].reshape(ROLL, NF, -1), -1)
        f = lambda k: round(float((k - 273.15) * 1.8 + 32), 1)
        return [{'valid': (t0 + dt.timedelta(hours=k + 1)).strftime('%Y-%m-%dT%H:00Z'), 't': f(m[k, 0]), 'td': f(m[k, 1]),
                 'w': round(float(math.hypot(m[k, 2], m[k, 3]) * 2.237), 1), 'g': round(float(max(0.0, m[k, 4]) * 2.237), 1),
                 'c': round(float(min(100.0, max(0.0, m[k, 5])))), 'lead': k + 1} for k in range(ROLL)]
    issued = t0.strftime('%Y-%m-%dT%H:00Z')
    if TEST:
        print(f'Test ({model}), nothing sent:')
        for L in at(33.81, -84.17)[:6] + at(33.81, -84.17)[8::3]: print('  Stone Mountain', L)
        return 0
    if not API or not KEY: print('Missing the STORMWATCH_API / STORMWATCH_KEY secrets.'); return 1
    H = {'X-StormWatch-Key': KEY}
    places = json.loads(get(API + '/api/places', headers=H) or b'{}').get('places', [])
    if BACKUP and places:
        try:
            g = json.loads(get(f"{API}/api/gnow?lat={places[0]['lat']}&lon={places[0]['lon']}", headers=H) or b'{}')
            last = (g.get('latest') or {}).get('at')
            if last and (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(last.replace('Z', '+00:00'))).total_seconds() < 3600:
                print(f'The main run already sent a forecast at {last}; nothing to do.'); return 0
        except Exception as e: print('(could not check the main run:', e, ')')
    runs = [{'lat': p['lat'], 'lon': p['lon'], 'leads': L} for p in places if (L := at(p['lat'], p['lon']))]
    if not runs: print(f'None of the {len(places)} place(s) is on the Georgia Now map; nothing to send.'); return 0
    req = urllib.request.Request(API + '/api/gnow', data=json.dumps({'issued': issued, 'model': model, 'runs': runs}).encode(), method='POST',
                                 headers=dict(UA, **H, **{'Content-Type': 'application/json'}))
    with urllib.request.urlopen(req, timeout=60) as r: res = json.loads(r.read() or b'{}')
    print(f'Sent to your server ({model}): {len(runs)} of {len(places)} place(s) ({res.get("saved", "?")} saved).')
    return 0

if __name__ == '__main__':
    sys.exit(main())
