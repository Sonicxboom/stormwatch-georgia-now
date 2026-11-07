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

StormWatchNow+: when SW_NOW_PLUS_MODEL points to a StormWatchNow+ checkpoint (and SW_HRRR_REGRID to the HRRR -> map
cell matching saved by hrrr_archive.py), the HRRR's newest complete run is downloaded too (NOAA's archive on AWS), cut to
the same map, and StormWatchNow+ corrects it. StormWatchNow+ is used from 3 hours ahead and StormWatchNow for 1-2 hours
(each won those hours in the airport test). The HRRR run at the map's hour isn't finished at :55, so the run before it
is used, matched by valid time (its 18-hour forecast stands in for the 18th hour).
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

_LIVE = {}   # this hour's map forecasts by source, for the live airport test (SW_LIVE_TEST)
HRRR_BUCKET = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com'
HRRR_FIELDS = [('TMP', '2 m above ground'), ('DPT', '2 m above ground'), ('UGRD', '10 m above ground'),
               ('VGRD', '10 m above ground'), ('GUST', 'surface'), ('TCDC', 'entire atmosphere')]   # same order as the maps

def fetch_hrrr(t0, leads, regrid):
    """The HRRR's forecast for t0 + each lead, from its newest run that has them all: (leads, NF, H, W) real units, or None."""
    from concurrent.futures import ThreadPoolExecutor
    import pygrib
    z = np.load(regrid); R, C = z['r'], z['c']
    for back in (0, 1, 2):
        init = t0 - dt.timedelta(hours=back)
        fh = [min(18, L + back) for L in leads]
        url = lambda f: f'{HRRR_BUCKET}/hrrr.{init:%Y%m%d}/conus/hrrr.t{init:%H}z.wrfsfcf{f:02d}.grib2'
        idx = {f: get(url(f) + '.idx') for f in sorted(set(fh))}
        if any(v is None for v in idx.values()): continue
        def one(job):
            f, w = job
            lines = [l.split(':') for l in idx[f].decode().strip().splitlines()]
            for i, c in enumerate(lines):
                if c[3] == w[0] and c[4] == w[1]:
                    b = get(url(f), (int(c[1]), int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ''))
                    if not b: return None
                    v = pygrib.fromstring(b).values
                    v = np.ma.filled(v.astype(np.float32), np.nan) if np.ma.isMaskedArray(v) else v.astype(np.float32)
                    return v[R, C]
            return None
        jobs = [(f, w) for f in fh for w in HRRR_FIELDS]
        with ThreadPoolExecutor(8) as ex: got = list(ex.map(one, jobs))
        if any(g is None for g in got): continue
        print(f'HRRR: the {init:%H}z run{" (its 18-hour forecast stands in for the last hour)" if back and max(leads) + back > 18 else ""}', flush=True)
        return np.stack(got).reshape(len(leads), NF, SIZE, SIZE)
    print('(no complete HRRR run yet; StormWatchNow+ skipped this hour)'); return None

# NOAA's RRFS: the 3 km model that replaced the NAM (and HREF) on Nov 3, 2026, built on different code than the HRRR, so it
# misses in different ways. Same grid as the HRRR, so the HRRR's map-cell matching (SW_HRRR_REGRID) cuts it to the map.
# Every field, every hour 1-18: for your places and the next-18-hours scoreboard, for the Map AI's training archive
# (SW_RRFS_SAVE), and as an input to StormWatchNow+ once it's been trained to read the RRFS.
RRFS_BUCKET = 'https://noaa-rrfs-ops-pds.s3.amazonaws.com'
RRFS_FIELDS = [('TMP', '2 m above ground'), ('DPT', '2 m above ground'), ('UGRD', '10 m above ground'),
               ('VGRD', '10 m above ground'), ('GUST', 'surface'), ('TCDC', 'entire atmosphere (considered as a single layer)')]
MAP_LEADS = [1, 2, 3, 6, 9, 12, 15, 18]   # the Map AI's hours ahead

def fetch_rrfs(now, regrid):
    """The RRFS's newest complete run: (init time, (18, NF, H, W) real units for 1-18 hours after it), or (None, None)."""
    from concurrent.futures import ThreadPoolExecutor
    import pygrib
    if not regrid or not os.path.exists(regrid): return None, None
    z = np.load(regrid); R, C = z['r'], z['c']
    for back in range(1, 5):   # a run takes ~1.5 hours to finish
        init = now - dt.timedelta(hours=back)
        url = lambda f: f'{RRFS_BUCKET}/rrfs.{init:%Y%m%d}/{init:%H}/rrfs.t{init:%H}z.2dfld.3km.f{f:03d}.conus.grib2'
        idx = {f: get(url(f) + '.idx') for f in range(1, 19)}
        if any(v is None for v in idx.values()): continue
        def one(job):
            f, k = job; w = RRFS_FIELDS[k]
            lines = [l.split(':') for l in idx[f].decode().strip().splitlines()]
            for i, c in enumerate(lines):   # the instant value ("12 hour fcst"), not the hour's average or max
                if len(c) > 5 and c[3] == w[0] and c[4] == w[1] and c[5].endswith('hour fcst') and '-' not in c[5]:
                    b = get(url(f), (int(c[1]), int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ''))
                    if not b: return None
                    v = pygrib.fromstring(b).values
                    v = np.ma.filled(v.astype(np.float32), np.nan) if np.ma.isMaskedArray(v) else v.astype(np.float32)
                    return v[R, C]
            return None
        jobs = [(f, k) for f in range(1, 19) for k in range(NF)]
        with ThreadPoolExecutor(12) as ex: got = list(ex.map(one, jobs))
        if any(g is None for g in got): continue
        X = np.full((18, NF, SIZE, SIZE), np.nan, np.float32)
        for (f, k), g in zip(jobs, got): X[f - 1, k] = g
        print(f'RRFS: the {init:%H}z run', flush=True)
        return init, X
    print('(no complete RRFS run found; skipped this hour)'); return None, None

def save_rrfs(folder, init, X):
    """The RRFS's Georgia maps at the Map AI's hours, for training (same layout as hrrr_archive.py's files)."""
    if not folder or X is None: return
    os.makedirs(folder, exist_ok=True); f = os.path.join(folder, f'{init:%Y%m%d%H}.npz')
    if os.path.exists(f): return
    np.savez_compressed(f, x=X[[L - 1 for L in MAP_LEADS]].astype(np.float16), leads=np.array(MAP_LEADS), ok=np.ones(len(MAP_LEADS), bool))

def now_forecast(path, t0, cur, st, plus_path='', regrid='', rrfs=(None, None)):
    """StormWatchNow (and StormWatchNow+ from 3 hours ahead, when it's set up): (18, NF, H, W) in real units from the last
    24 hours of maps, or None (then Georgia Now runs). Returns (forecast, name)."""
    import torch
    def load(p):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        net = build_net(ck['cin'], len(ck['leads']) * ck['nf'], w=ck['width'], levels=ck['levels']); net.load_state_dict(ck['state']); net.eval()
        return ck, net
    cks = {}
    if path and os.path.exists(path): cks['now'] = load(path)
    if plus_path and os.path.exists(plus_path) and regrid and os.path.exists(regrid): cks['plus'] = load(plus_path)
    if not cks: return None, ''
    ck0 = next(iter(cks.values()))[0]
    SC = np.array(ck0['scale'], np.float32); CE = tuple(ck0['center']); LEADS, HIST = ck0['leads'], ck0['hist']
    maps = []
    for h in HIST:
        m = cur if h == 0 else fetch_hour(t0 - dt.timedelta(hours=h), st)
        if m is None: print(f'(the map from {h} h ago is missing; using Georgia Now this hour)'); return None, ''
        maps.append(m)
    sc = lambda a: (a - SC[:, 0, None, None]) / SC[:, 1, None, None]
    M = [torch.nan_to_num(torch.tensor(sc(m)))[None] for m in maps]
    S = torch.tensor(np.stack([st['hgt'].astype(np.float32) / 1000.0, (st['lat'] - CE[0]) / 3.0, (st['lon'] - CE[1]) / 3.0]).astype(np.float32))
    tc = torch.tensor(time_ch(t0))[None, :, None, None].expand(-1, -1, SIZE, SIZE)
    real = lambda P: P * SC[None, :, 1, None, None] + SC[None, :, 0, None, None]
    out_by = {}
    if 'now' in cks:
        with torch.no_grad(): d = cks['now'][1](torch.cat(M + [S[None], tc], 1)).view(len(LEADS), NF, SIZE, SIZE)
        out_by['now'] = real((M[0][0][None] + d).numpy())
    if 'plus' in cks:
        try:
            Hr = fetch_hrrr(t0, LEADS, regrid)
            if Hr is not None: _LIVE['hrrr'] = Hr
            if Hr is not None:
                hs = torch.nan_to_num(torch.tensor(sc(Hr).astype(np.float32)))   # (leads, NF, H, W), scaled like the maps
                extra = []
                if cks['plus'][0].get('rrfs'):   # trained to read the RRFS too: its newest run, matched by valid time, + the flag
                    r_init, RX = rrfs
                    if RX is not None:
                        back = int((t0 - r_init).total_seconds() // 3600)
                        Rm = np.stack([RX[min(18, L + back) - 1] for L in LEADS])   # its 18-hour forecast stands in past its end
                        ok = bool(np.isfinite(Rm).all())
                    else: ok = False
                    rr = torch.nan_to_num(torch.tensor(sc(Rm).astype(np.float32))) if ok else torch.zeros(len(LEADS), NF, SIZE, SIZE)
                    extra = [rr.reshape(1, len(LEADS) * NF, SIZE, SIZE), torch.full((1, 1, SIZE, SIZE), 1.0 if ok else 0.0)]
                    print(f'StormWatchNow+ reads the RRFS ({r_init:%H}z run)' if ok else '(StormWatchNow+: no RRFS this hour; it runs without it)', flush=True)
                with torch.no_grad(): d = cks['plus'][1](torch.cat(M + [S[None], tc, hs.reshape(1, len(LEADS) * NF, SIZE, SIZE)] + extra, 1)).view(len(LEADS), NF, SIZE, SIZE)
                out_by['plus'] = real((hs + d).numpy())
        except Exception as e: print('(StormWatchNow+ failed:', e, ')')
    if not out_by: return None, ''
    # StormWatchNow for 1-2 hours ahead, StormWatchNow+ from 3 hours (each won those hours in the airport test)
    # (at 2 hours, where they tied, the two are averaged so the hand-off has no jump)
    _LIVE.update(out_by); _LIVE['leads'] = list(LEADS)
    pick = lambda i, L: out_by['now'][i] if 'plus' not in out_by or L <= 1 else out_by['plus'][i] if 'now' not in out_by or L >= 3 else (out_by['now'][i] + out_by['plus'][i]) / 2
    P = np.stack([pick(i, L) for i, L in enumerate(LEADS)])
    name = 'StormWatchNow + StormWatchNow+' if len(out_by) == 2 else 'StormWatchNow+' if 'plus' in out_by else 'StormWatchNow'
    xs, out = [0] + list(LEADS), []
    for k in range(1, ROLL + 1):                                               # straight line between the forecast hours
        j = next(i for i in range(1, len(xs)) if xs[i] >= k); a, b = xs[j - 1], xs[j]; w = (k - a) / (b - a)
        lo = cur if a == 0 else P[j - 2]; hi = P[j - 1]
        out.append(hi if w == 1 else lo * (1 - w) + hi * w)
    _LIVE['map_ai'] = np.stack([P[i] for i in range(len(LEADS))])   # what the hourly sent, at the forecast hours
    return np.stack(out).astype(np.float32), name

def save_live_test(folder, t0, model):
    """The live airport test: this hour's forecasts (Map AI as sent, StormWatchNow, StormWatchNow+, the HRRR) at every
    airport on the map, saved for the scoring job to check once the hours have passed. Nothing here changes the forecast."""
    stf = os.environ.get('SW_STATIONS', '')
    if not folder or not _LIVE.get('leads') or not stf or not os.path.exists(stf): return
    z = np.load(stf, allow_pickle=True); R, C = np.asarray(z['r']), np.asarray(z['c'])
    def at(A):   # (leads, NF, H, W) real units -> (leads, stations, 3): temp F, dew F, wind mph
        a = A[:, :, R, C]
        return np.stack([(a[:, 0] - 273.15) * 1.8 + 32, (a[:, 1] - 273.15) * 1.8 + 32, np.hypot(a[:, 2], a[:, 3]) * 2.237], -1).astype(np.float32)
    srcs = {k: at(_LIVE[k]) for k in ('map_ai', 'now', 'plus', 'hrrr') if k in _LIVE}
    os.makedirs(os.path.join(folder, 'runs'), exist_ok=True)
    np.savez_compressed(os.path.join(folder, 'runs', f'{t0:%Y%m%d%H}.npz'), issued=t0.isoformat(), model=model, leads=np.array(_LIVE['leads']),
                        sid=np.asarray(z['sid']), **srcs)
    print(f'Live airport test: saved {", ".join(srcs)} at {len(R)} airports', flush=True)

def main():
    import torch
    torch.set_num_threads(os.cpu_count() or 2)
    ck = torch.load(MODEL, map_location='cpu', weights_only=False); st = np.load(STATIC)
    SCALE = np.array(ck['scale'], np.float32); CENTER = tuple(ck['center'])
    net = build_net(ck['cin'], ck['nf']); net.load_state_dict(ck['state']); net.eval()
    S = torch.tensor(np.stack([st['hgt'].astype(np.float32) / 1000.0, (st['lat'] - CENTER[0]) / 3.0, (st['lon'] - CENTER[1]) / 3.0]).astype(np.float32))
    NOW, NOWP, REGRID = os.environ.get('SW_NOW_MODEL', ''), os.environ.get('SW_NOW_PLUS_MODEL', ''), os.environ.get('SW_HRRR_REGRID', '')
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
    # NOAA's RRFS (scored next to the others, used by your hourly where it wins, and read by StormWatchNow+ once trained to)
    r_init = RX = None
    try: r_init, RX = fetch_rrfs(now, REGRID)
    except Exception as e: print('(RRFS failed:', e, ')')
    tt, F, model = time.time(), None, 'Georgia Now'
    if (NOW and os.path.exists(NOW)) or (NOWP and os.path.exists(NOWP)):
        try: F, nm = now_forecast(NOW, t0, cur, st, NOWP, REGRID, (r_init, RX))
        except Exception as e: F = None; print('(StormWatchNow failed:', e, '; using Georgia Now)')
        if F is not None: model = nm
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
    if not TEST:
        try: save_live_test(os.environ.get('SW_LIVE_TEST', ''), t0, model)
        except Exception as e: print('(live airport test not saved:', e, ')')
    if not TEST:
        try: save_rrfs(os.environ.get('SW_RRFS_SAVE', ''), r_init, RX)
        except Exception as e: print('(RRFS maps not saved:', e, ')')
    if not TEST and os.environ.get('SW_MAPAI_SAVE'):
        # Map AI's "right now" map (its 1-hour forecast: temperature, dew point, wind), kept so StormWatch's 3D-Var can start
        # from it instead of the raw HRRR. Small (4 fields, half precision); the last 60 days are kept.
        try:
            d = os.environ['SW_MAPAI_SAVE']; os.makedirs(d, exist_ok=True); v = t0 + dt.timedelta(hours=1)
            np.savez_compressed(f'{d}/mapai_{v:%Y%m%d%H}.npz', valid=v.isoformat(), model=model, x=F[0, :4].astype(np.float16))
            old = sorted(f for f in os.listdir(d) if f.startswith('mapai_'))[:-60 * 24]
            for f in old: os.remove(os.path.join(d, f))
        except Exception as e: print('(Map AI map not saved:', e, ')')
    def at(la, lo, A=None, base=None):   # 3x3 mean around a spot -> leads, or None if it's off the map
        A = F if A is None else A; base = t0 if base is None else base
        d = (st['lat'] - la) ** 2 + (((st['lon'] - lo + 180) % 360 - 180) * math.cos(math.radians(la))) ** 2
        r, c = (int(v) for v in np.unravel_index(np.argmin(d), d.shape))
        if d[r, c] > 0.03 ** 2 or not (8 <= r < SIZE - 8 and 8 <= c < SIZE - 8): return None
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); m = np.nanmean(A[:, :, r - 1:r + 2, c - 1:c + 2].reshape(len(A), NF, -1), -1)
        f = lambda k: round(float((k - 273.15) * 1.8 + 32), 1) if np.isfinite(k) else None
        n = lambda v, s=1: round(float(v) * s, 1) if np.isfinite(v) else None
        return [{'valid': (base + dt.timedelta(hours=k + 1)).strftime('%Y-%m-%dT%H:00Z'), 't': f(m[k, 0]), 'td': f(m[k, 1]),
                 'w': n(math.hypot(m[k, 2], m[k, 3]), 2.237), 'g': n(max(0.0, m[k, 4]), 2.237) if np.isfinite(m[k, 4]) else None,
                 'c': round(float(min(100.0, max(0.0, m[k, 5])))) if np.isfinite(m[k, 5]) else None, 'lead': k + 1} for k in range(len(A))]
    def at_rrfs(la, lo):   # the RRFS at a place: temperature, dew point and wind for the hours after this run's map
        if RX is None: return None
        L = at(la, lo, RX, r_init)
        return [{k: x[k] for k in ('valid', 't', 'td', 'w')} for x in L if x['t'] is not None and x['valid'] > t0.strftime('%Y-%m-%dT%H:00Z')] if L else None
    issued = t0.strftime('%Y-%m-%dT%H:00Z')
    if TEST:
        print(f'Test ({model}), nothing sent:')
        for L in at(33.81, -84.17)[:6] + at(33.81, -84.17)[8::3]: print('  Stone Mountain', L)
        R_ = at_rrfs(33.81, -84.17)
        if R_: print(f'  RRFS ({r_init:%H}z run), Stone Mountain:', R_[0], '...', R_[-1])
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
    runs = [{'lat': p['lat'], 'lon': p['lon'], 'leads': L, **({'rrfs': Rr} if (Rr := at_rrfs(p['lat'], p['lon'])) else {})}
            for p in places if (L := at(p['lat'], p['lon']))]
    if not runs: print(f'None of the {len(places)} place(s) is on the Georgia Now map; nothing to send.'); return 0
    req = urllib.request.Request(API + '/api/gnow', data=json.dumps({'issued': issued, 'model': model, 'runs': runs}).encode(), method='POST',
                                 headers=dict(UA, **H, **{'Content-Type': 'application/json'}))
    with urllib.request.urlopen(req, timeout=60) as r: res = json.loads(r.read() or b'{}')
    print(f'Sent to your server ({model}{", with the RRFS" if RX is not None else ""}): {len(runs)} of {len(places)} place(s) ({res.get("saved", "?")} saved).')
    return 0

LONG_FIELDS = [('TMP', '2 m above ground'), ('DPT', '2 m above ground'), ('UGRD', '10 m above ground'), ('VGRD', '10 m above ground'),
               ('GUST', 'surface'), ('TCDC', 'entire atmosphere (considered as a single layer)'), ('APCP', 'surface')]

def rrfs_long():
    """The RRFS's newest 84-hour run (it makes one at 00, 06, 12 and 18 UTC) at each of your places, hour by hour, for the
    7-day blend and the daily scoreboard (it took over from the NAM). Sent to your server as rows like Tomorrow.io's:
    [unix time, temp F, feels-like (none), dew point F, humidity %, wind mph, gust mph, cloud %, rain chance (none), rain in]."""
    from concurrent.futures import ThreadPoolExecutor
    import pygrib
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    init = None
    for back in range(1, 19):
        t = now - dt.timedelta(hours=back)
        if t.hour % 6: continue
        u = f'{RRFS_BUCKET}/rrfs.{t:%Y%m%d}/{t:%H}/rrfs.t{t:%H}z.2dfld.3km.f084.conus.grib2.idx'
        if get(u) is not None: init = t; break
    if init is None: print('No finished 84-hour RRFS run yet.'); return 0
    url = lambda f: f'{RRFS_BUCKET}/rrfs.{init:%Y%m%d}/{init:%H}/rrfs.t{init:%H}z.2dfld.3km.f{f:03d}.conus.grib2'
    H = {'X-StormWatch-Key': KEY}
    if not TEST and (not API or not KEY): print('Missing the STORMWATCH_API / STORMWATCH_KEY secrets.'); return 1
    places = [(33.81, -84.17)] if TEST else [(p['lat'], p['lon']) for p in json.loads(get(API + '/api/places', headers=H) or b'{}').get('places', [])]
    if not places: print('No places.'); return 0
    def field(f, w, idx):
        lines = [l.split(':') for l in idx.strip().splitlines()]
        for i, c in enumerate(lines):
            if len(c) < 6 or c[3] != w[0] or c[4] != w[1]: continue
            if w[0] == 'APCP' and c[5] != f'{f - 1}-{f} hour acc fcst': continue   # the rain in that one hour
            if w[0] != 'APCP' and '-' in c[5]: continue                            # the value at that hour, not an average
            b = get(url(f), (int(c[1]), int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ''))
            return pygrib.fromstring(b) if b else None
        return None
    # each place's grid point, from the grid's own lat/lon (the same grid as the HRRR, ~1.9 million points)
    m0 = field(1, LONG_FIELDS[0], get(url(1) + '.idx').decode())
    la, lo = m0.latlons(); lo = (lo + 180) % 360 - 180
    pts = []
    for (pla, plo) in places:
        d = (la - pla) ** 2 + ((lo - plo) * math.cos(math.radians(pla))) ** 2
        r, c = np.unravel_index(np.argmin(d), d.shape); pts.append((int(r), int(c)) if d[r, c] < 0.05 ** 2 else None)
    def one(f):
        idx = get(url(f) + '.idx')
        if not idx: return f, None
        idx = idx.decode(); out = {}
        for w in LONG_FIELDS:
            m = field(f, w, idx)
            if m is None: return f, None
            v = m.values; out[w[0]] = [float(v[p]) if p else None for p in pts]
        return f, out
    tt = time.time()
    with ThreadPoolExecutor(16) as ex: got = dict(ex.map(one, range(1, 85)))
    ok = [f for f in range(1, 85) if got.get(f)]
    print(f'RRFS {init:%Y-%m-%d %H}z 84-hour run: {len(ok)}/84 hours, {len(places)} place(s), {time.time() - tt:.0f} s', flush=True)
    if len(ok) < 70: print('Too many hours missing; not sent.'); return 0
    runs = []
    for k, (pla, plo) in enumerate(places):
        if pts[k] is None: continue
        rows = []
        for f in ok:
            g = got[f]; tK, dK = g['TMP'][k], g['DPT'][k]
            tF, dF = (tK - 273.15) * 1.8 + 32, (dK - 273.15) * 1.8 + 32
            tc, dc = tK - 273.15, dK - 273.15
            rh = 100 * math.exp(17.625 * dc / (243.04 + dc)) / math.exp(17.625 * tc / (243.04 + tc))
            rows.append([int((init + dt.timedelta(hours=f)).replace(tzinfo=dt.timezone.utc).timestamp()), round(tF, 1), None, round(dF, 1), round(min(100, rh)),
                         round(math.hypot(g['UGRD'][k], g['VGRD'][k]) * 2.237, 1), round(max(0.0, g['GUST'][k]) * 2.237, 1),
                         round(min(100.0, max(0.0, g['TCDC'][k]))), None, round(max(0.0, g['APCP'][k]) / 25.4, 3)])
        runs.append({'lat': pla, 'lon': plo, 'h': rows})
    if TEST:
        r = runs[0]['h']; print('Stone Mountain:', r[0], '...', r[-1]); return 0
    req = urllib.request.Request(API + '/api/rrfs', data=json.dumps({'issued': init.strftime('%Y-%m-%dT%H:00Z'), 'runs': runs}).encode(), method='POST',
                                 headers=dict(UA, **H, **{'Content-Type': 'application/json'}))
    with urllib.request.urlopen(req, timeout=60) as r: res = json.loads(r.read() or b'{}')
    print(f'Sent to your server: {len(runs)} place(s) ({res.get("saved", "?")} saved).')
    return 0


# ---- NOAA AIGEFS: NOAA's AI ensemble, a second AI opinion for the 7-day highs/lows (and StormWatch Long's training data)
EAGLE = 'https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/EAGLE_ensemble'
AIG_MEMBERS = 10                        # 10 of its 31 members: their average is nearly the full average, at a third the download
AIG_LEADS = list(range(6, 169, 6))      # 7 days, every 6 hours
AIG_SFC = [('UGRD', '10 m above ground'), ('VGRD', '10 m above ground'), ('TMP', '2 m above ground'), ('PRES', 'surface'),
           ('PRMSL', 'mean sea level'), ('APCP', 'surface')]
AIG_PRES = [('SPFH', '1000 mb'), ('SPFH', '925 mb')]
AIG_BOX = (29.0, 36.5, -89.0, -78.0)   # Georgia and around it (lat, lat, lon, lon), kept for training

def aigefs_url(init, m, f, kind):
    return f'{EAGLE}/aigefs.{init:%Y%m%d}/{init:%H}/mem{m:03d}/model/atmos/grib2/aigefs.t{init:%H}z.{kind}.f{f:03d}.grib2'

def aigefs_grab(init, members=AIG_MEMBERS, leads=AIG_LEADS, workers=24):
    """The Georgia box of each member's fields, every 6 hours. Returns (lat 1-D, lon 1-D, {field: array[member, lead, y, x]})
    with dew point (K) worked out from the humidity just above the ground. Only the needed pieces of each file are fetched."""
    from concurrent.futures import ThreadPoolExecutor
    import pygrib
    box = {}
    def msgs(m, f, kind, want):
        idx = get(aigefs_url(init, m, f, kind) + '.idx')
        if not idx: return None
        lines = [l.split(':') for l in idx.decode().strip().splitlines()]; out = {}
        for i, c in enumerate(lines):
            for w in want:
                if w in out or len(c) < 6 or c[3] != w[0] or c[4] != w[1]: continue
                if w[0] == 'APCP' and c[5] != f'{f - 6}-{f} hour acc fcst': continue   # the rain in those 6 hours
                b = get(aigefs_url(init, m, f, kind), (int(c[1]), int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ''))
                if b: out[w] = pygrib.fromstring(b)
        return out if len(out) == len(want) else None
    def one(job):
        m, f = job
        a, b = msgs(m, f, 'sfc', AIG_SFC), msgs(m, f, 'pres', AIG_PRES)
        if not a or not b: return job, None
        if 'ij' not in box:
            la, lo = a[AIG_SFC[0]].latlons(); lat, lon = la[:, 0], (lo[0] + 180) % 360 - 180
            ys = np.where((lat >= AIG_BOX[0]) & (lat <= AIG_BOX[1]))[0]; xs = np.where((lon >= AIG_BOX[2]) & (lon <= AIG_BOX[3]))[0]
            box.update(ij=(ys, xs), lat=lat[ys], lon=lon[xs])
        ys, xs = box['ij']; cut = lambda g: np.asarray(g.values, 'f4')[np.ix_(ys, xs)]
        v = {w[0] if w[0] != 'TMP' else 'T': cut(g) for w, g in a.items()}
        q1, q9 = cut(b[AIG_PRES[0]]), cut(b[AIG_PRES[1]]); ps = v['PRES'] / 100
        q = q9 + (q1 - q9) * np.clip((ps - 925) / 75, 0, 1)                     # humidity at the ground's pressure
        e = np.maximum(q * ps / (0.622 + 0.378 * q), 1e-3); g_ = np.log(e / 6.112)
        v['Td'] = np.minimum(243.5 * g_ / (17.67 - g_) + 273.15, v['T'])
        v['WS'] = np.hypot(v['UGRD'], v['VGRD'])
        return job, v
    jobs = [(m, f) for m in range(members) for f in leads]
    with ThreadPoolExecutor(workers) as ex: got = dict(ex.map(one, jobs))
    if 'ij' not in box: return None
    names = ['T', 'Td', 'WS', 'UGRD', 'VGRD', 'PRMSL', 'APCP']
    ny, nx = len(box['lat']), len(box['lon']); F = {k: np.full((members, len(leads), ny, nx), np.nan, 'f4') for k in names}
    for (m, f), v in got.items():
        if v is None: continue
        for k in names: F[k][m, leads.index(f)] = v[k]
    return box['lat'], box['lon'], F

def aigefs_newest(back=36):
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    for h in range(0, back, 1):
        t = now - dt.timedelta(hours=h)
        if t.hour % 6: continue
        if get(aigefs_url(t, AIG_MEMBERS - 1, AIG_LEADS[-1], 'sfc') + '.idx') is not None: return t
    return None

# ---- StormWatch Long: your own correction of AIGEFS's daily highs and lows (trained on Kaggle, tools/kaggle_long_train.py in
# the StormWatch repo, ~150 airports since June 2025). Run on each member, averaged ("ensemble MOS"). Used per field only
# where its honest test (the last 60 days, never trained on) was at least as good as AIGEFS with a bias fix.
def long_model(root):
    """The newest StormWatch Long: live/ if one was promoted, else the newest training run. Loaded with the same
    scikit-learn version it was trained with (installed on the spot if this machine has a different one)."""
    import glob, re, pickle, subprocess
    cand = [f'{root}/live/stormwatch_long.pkl'] if os.path.exists(f'{root}/live/stormwatch_long.pkl') else sorted(glob.glob(f'{root}/runs/*-long/stormwatch_long.pkl'))[-1:]
    if not cand: return None
    blob = open(cand[0], 'rb').read()
    v = re.search(rb'_sklearn_version.{1,6}?(\d+\.\d+(?:\.\d+)?)', blob, re.S)
    try:
        import sklearn; have = sklearn.__version__
    except ImportError: have = None
    if v and have != v.group(1).decode():
        tgt = '/tmp/skl'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--target', tgt, f'scikit-learn=={v.group(1).decode()}'], check=True)
        sys.path.insert(0, tgt)
        for k in [k for k in sys.modules if k == 'sklearn' or k.startswith('sklearn.')]: del sys.modules[k]
    M = pickle.loads(blob); rep = M.get('report', {})
    M['use'] = [f for f in ('high', 'low') if f in rep and rep[f]['stormwatch_long'] <= rep[f]['ai_plus_bias_fix']]
    M['file'] = cand[0]
    return M

def long_feats(a, k0, k, lat, lon, elev, day):
    """a: [T K, Td K, wind m/s, rain mm per 6 h][28 steps] for one member at one place; k0: the step at 06 UTC of that
    Eastern day; k: days ahead minus 1 (as in training)."""
    i = [k0, k0 + 1, k0 + 2, k0 + 3]
    T = (a[0, i] - 273.15) * 1.8 + 32; Td = (a[1, i] - 273.15) * 1.8 + 32
    nxt = (a[0, min(k0 + 4, a.shape[1] - 1)] - 273.15) * 1.8 + 32
    doy = day.timetuple().tm_yday
    return [T[0], T[1], T[2], T[3], T.max(), T.min(), nxt, Td[1], Td[2], T[2] - Td[2], a[2, i].mean() * 2.237,
            a[3, i].sum() / 25.4, k + 1, lat, lon, elev, math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25)]

def long_fix(M, init, mem, lat, lon, elev):
    """mem: [members, 4, 28 steps] at the place -> {Eastern date: {'high': F, 'low': F}} (only the fields in use).
    Each whole Eastern day (its 06, 12, 18 and 24 UTC steps all in the run) gets the model's answer, averaged over members."""
    out = {}
    for k0 in range(len(AIG_LEADS) - 3):
        v0 = init + dt.timedelta(hours=AIG_LEADS[k0])
        if v0.hour != 6: continue
        day = v0.date(); k = (day - init.date()).days
        if not 0 <= k <= 6: continue
        X = np.array([long_feats(m, k0, k, lat, lon, elev, day) for m in mem], 'f4')
        if not np.isfinite(X).all(): continue
        out[day] = {f: float(np.mean(X[:, 4 if f == 'high' else 5] + M['models'][f].predict(X))) for f in M['use']}
    return out

def eastern_day(ts):   # the Eastern-time calendar date of a UTC timestamp (EST/EDT by the US rule)
    t = dt.datetime.fromtimestamp(ts, dt.timezone.utc).replace(tzinfo=None); y = t.year
    s = dt.datetime(y, 3, 8 + (6 - dt.date(y, 3, 8).weekday()) % 7, 7); e = dt.datetime(y, 11, 1 + (6 - dt.date(y, 11, 1).weekday()) % 7, 6)
    return (t - dt.timedelta(hours=4 if s <= t < e else 5)).date()

def apply_long(rows, fix):
    """Reshape each Eastern day's hourly temperatures so the day's high and low land on StormWatch Long's (the server only
    takes highs and lows from AIGEFS, never its hours). Dew point is kept at or below the new temperature."""
    by = {}
    for r in rows: by.setdefault(eastern_day(r[0]), []).append(r)
    for day, rs in by.items():
        f = fix.get(day)
        if not f or len(rs) < 20: continue
        T = [r[1] for r in rs]; hi, lo = max(T), min(T)
        nh, nl = f.get('high', hi), f.get('low', lo)
        if nh < nl + 1: nh, nl = (nh + nl) / 2 + 0.5, (nh + nl) / 2 - 0.5
        for r in rs:
            t = r[1]; r[1] = round(nl + (t - lo) * (nh - nl) / (hi - lo), 1) if hi - lo > 0.5 else round(t + (nh + nl - hi - lo) / 2, 1)
            if r[3] is not None: r[3] = min(r[3], r[1])
    return rows

def elevation(lat, lon):   # meters, as in training (IEM's airport list)
    try: return float(json.loads(get(f'https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}'))['elevation'][0])
    except Exception: return 300.0

def aigefs_long():
    """NOAA AIGEFS at your places for the 7-day blend, plus its Georgia box saved for training StormWatch Long."""
    init = aigefs_newest()
    if init is None: print('No finished AIGEFS run in the last 36 hours.'); return 0
    H = {'X-StormWatch-Key': KEY}
    if not TEST and (not API or not KEY): print('Missing the STORMWATCH_API / STORMWATCH_KEY secrets.'); return 1
    tt = time.time(); res = aigefs_grab(init)
    if res is None: print('AIGEFS download failed.'); return 1
    lat, lon, F = res
    have = np.isfinite(F['T']).all(axis=(2, 3))                                  # [member, lead]
    ok_m = np.where(have.mean(1) > 0.9)[0]
    print(f'AIGEFS {init:%Y-%m-%d %H}z: {len(ok_m)}/{AIG_MEMBERS} members, {len(AIG_LEADS)} times, {time.time() - tt:.0f} s', flush=True)
    if len(ok_m) < 6: print('Too few members; not sent.'); return 0
    save = os.environ.get('SW_AIGEFS_SAVE')
    if save:   # the average and spread over Georgia, for training and running StormWatch Long
        os.makedirs(save, exist_ok=True)
        np.savez_compressed(f'{save}/aigefs_{init:%Y%m%d%H}.npz', lat=lat, lon=lon, leads=np.array(AIG_LEADS), members=len(ok_m),
                            **{f'{k}_mean': np.nanmean(v[ok_m], 0) for k, v in F.items()},
                            **{f'{k}_sd': np.nanstd(v[ok_m], 0) for k, v in F.items() if k in ('T', 'Td', 'WS', 'APCP')},
                            **{f'{k}_members': v[ok_m].astype('f2') for k, v in F.items() if k in ('T', 'Td', 'WS', 'APCP')})   # StormWatch Long runs on each
    M = None
    if os.environ.get('SW_LONG_DIR'):
        try:
            M = long_model(os.environ['SW_LONG_DIR'])
            if not M: print('StormWatch Long: none trained yet', flush=True)
            else: print(f'StormWatch Long: {M["file"]} ({M.get("run")}) for ' + (' and '.join(f + 's' for f in M['use']) if M['use'] else 'nothing (not better than a bias fix)'), flush=True)
            if M and not M['use']: M = None
        except Exception as e: print(f'StormWatch Long not used ({type(e).__name__}: {e})', flush=True); M = None
    places = [(33.81, -84.17)] if TEST else [(p['lat'], p['lon']) for p in json.loads(get(API + '/api/places', headers=H) or b'{}').get('places', [])]
    def at(a, pla, plo):   # bilinear, on the regular 0.25-degree grid
        fy = np.interp(pla, lat[::-1], np.arange(len(lat))[::-1]) if lat[0] > lat[-1] else np.interp(pla, lat, np.arange(len(lat)))
        fx = np.interp(plo, lon, np.arange(len(lon))); y0, x0 = int(fy), int(fx); y1, x1 = min(y0 + 1, len(lat) - 1), min(x0 + 1, len(lon) - 1)
        wy, wx = fy - y0, fx - x0
        return (a[..., y0, x0] * (1 - wy) * (1 - wx) + a[..., y0, x1] * (1 - wy) * wx + a[..., y1, x0] * wy * (1 - wx) + a[..., y1, x1] * wy * wx)
    runs = []
    for pla, plo in places:
        if not (AIG_BOX[0] < pla < AIG_BOX[1] and AIG_BOX[2] < plo < AIG_BOX[3]): continue
        mean = {k: np.nanmean(at(F[k][ok_m], pla, plo), 0) for k in ('T', 'Td', 'WS', 'APCP')}   # [lead]
        hrs = np.arange(AIG_LEADS[0], AIG_LEADS[-1] + 1)                     # filled in hour by hour between its 6-hour steps
        T, Td, WS = (np.interp(hrs, AIG_LEADS, mean[k]) for k in ('T', 'Td', 'WS'))
        rain = {f: max(0.0, float(mean['APCP'][i])) / 6 for i, f in enumerate(AIG_LEADS)}
        rows = []
        for i, hr in enumerate(hrs):
            tc, dc = T[i] - 273.15, Td[i] - 273.15
            rh = 100 * math.exp(17.625 * dc / (243.04 + dc)) / math.exp(17.625 * tc / (243.04 + tc))
            f6 = int(-(-hr // 6) * 6)
            rows.append([int((init + dt.timedelta(hours=int(hr))).replace(tzinfo=dt.timezone.utc).timestamp()), round(tc * 1.8 + 32, 1), None,
                         round(dc * 1.8 + 32, 1), round(min(100, rh)), round(float(WS[i]) * 2.237, 1), None, None, None,
                         round(rain.get(f6, 0.0) / 25.4, 3)])
        if M:
            try:
                mem = np.stack([at(F[k][ok_m], pla, plo) for k in ('T', 'Td', 'WS', 'APCP')], 1)   # [member, 4, step]
                fix = long_fix(M, init, mem, pla, plo, elevation(pla, plo))
                if TEST:
                    raw = {}
                    for r in rows: raw.setdefault(eastern_day(r[0]), []).append(r[1])
                    for d, f in sorted(fix.items()):
                        print(f'  {d}: AIGEFS high/low {max(raw.get(d, [0])):.0f}/{min(raw.get(d, [0])):.0f} -> StormWatch Long', ' '.join(f'{k} {v:.0f}' for k, v in f.items()))
                rows = apply_long(rows, fix)
            except Exception as e: print(f'StormWatch Long skipped at {pla},{plo}: {type(e).__name__}: {e}', flush=True)
        runs.append({'lat': pla, 'lon': plo, 'h': rows})
    if TEST:
        r = runs[0]['h']; print('Stone Mountain:', r[0], '...', r[-1], f'({len(r)} hours)'); return 0
    req = urllib.request.Request(API + '/api/aigefs', data=json.dumps({'issued': init.strftime('%Y-%m-%dT%H:00Z'), 'runs': runs}).encode(), method='POST',
                                 headers=dict(UA, **H, **{'Content-Type': 'application/json'}))
    with urllib.request.urlopen(req, timeout=60) as r: res = json.loads(r.read() or b'{}')
    print(f'Sent to your server: {len(runs)} place(s) ({res.get("saved", "?")} saved).')
    return 0

# ---- the crowd layer: every public Netatmo home station across Georgia, once an hour, saved for the 3D-Var (and later
# training). Netatmo only shows "right now", so these snapshots ARE the history. The Netatmo key never leaves your server:
# it lends this job a short-lived access token (/api/netatmo-token).
CROWD_BOX = (30.3, 35.1, -85.7, -80.8)
CROWD_TILE = 0.5            # degrees: ~100 requests an hour, well under Netatmo's 500 an hour

def crowd_parse(body, now_ts, max_age=3600):
    """Netatmo getpublicdata -> list of stations: id, lat, lon, alt (m), ts, tC, rh, p (hPa), wind/gust km/h, wind dir,
    rain mm in the last hour. Only readings from the last hour."""
    out = []
    for b in body or []:
        loc = (b.get('place') or {}).get('location')
        if not isinstance(loc, list) or len(loc) < 2: continue
        st = {'id': b.get('_id'), 'lat': float(loc[1]), 'lon': float(loc[0]), 'alt': (b.get('place') or {}).get('altitude')}
        for m in (b.get('measures') or {}).values():
            ty, res = m.get('type') or [], m.get('res')
            if res and ty:
                ts = max(int(k) for k in res); v = res[str(ts)] if str(ts) in res else res[ts]
                if now_ts - ts > max_age: continue
                for name, key in (('temperature', 'tC'), ('humidity', 'rh'), ('pressure', 'p')):
                    if name in ty and v[ty.index(name)] is not None: st[key] = float(v[ty.index(name)]); st.setdefault('ts', ts)
                if 'temperature' in ty: st['ts'] = ts
            if 'wind_strength' in m and now_ts - int(m.get('wind_timeutc') or 0) <= max_age:
                st.update(ws=m.get('wind_strength'), wd=m.get('wind_angle'), gust=m.get('gust_strength'))
            if 'rain_60min' in m and now_ts - int(m.get('rain_timeutc') or 0) <= max_age: st['rain1h'] = m.get('rain_60min')
        if 'tC' in st and -40 < st['tC'] < 55: out.append(st)
    return out

def crowd_snapshot():
    from concurrent.futures import ThreadPoolExecutor
    import gzip
    H = {'X-StormWatch-Key': KEY}
    if not API or not KEY: print('Missing the STORMWATCH_API / STORMWATCH_KEY secrets.'); return 1
    tok = json.loads(get(API + '/api/netatmo-token', headers=H) or b'{}').get('access')
    if not tok: print('No Netatmo token from your server (is Netatmo set up there?)'); return 0
    tiles = [(la, lo) for la in np.arange(CROWD_BOX[0], CROWD_BOX[1], CROWD_TILE) for lo in np.arange(CROWD_BOX[2], CROWD_BOX[3], CROWD_TILE)]
    now_ts = int(time.time())
    def one(t):
        la, lo = t
        u = (f'https://api.netatmo.com/api/getpublicdata?lat_ne={la + CROWD_TILE:.4f}&lon_ne={lo + CROWD_TILE:.4f}'
             f'&lat_sw={la:.4f}&lon_sw={lo:.4f}&required_data=temperature&filter=true')
        b = get(u, headers={'Authorization': f'Bearer {tok}'})
        try: return crowd_parse(json.loads(b or b'{}').get('body'), now_ts)
        except Exception: return []
    tt = time.time()
    with ThreadPoolExecutor(4) as ex: got = [x for xs in ex.map(one, tiles) for x in xs]
    seen, stations = set(), []
    for st in got:
        if st['id'] in seen: continue
        seen.add(st['id']); stations.append(st)
    print(f'Crowd layer: {len(stations)} Netatmo stations across Georgia ({len(tiles)} tiles, {time.time() - tt:.0f} s); '
          f'{sum("ws" in x for x in stations)} with wind, {sum("rain1h" in x for x in stations)} with rain', flush=True)
    d = os.environ.get('SW_CROWD_SAVE')
    if d and stations and not TEST:
        os.makedirs(d, exist_ok=True)
        t = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        with gzip.open(f'{d}/netatmo_{t:%Y%m%d%H%M}.json.gz', 'wt') as f: json.dump({'at': t.isoformat(timespec='minutes'), 'stations': stations}, f)
        old = sorted(x for x in os.listdir(d) if x.startswith('netatmo_'))[:-60 * 24]
        for x in old: os.remove(os.path.join(d, x))
    elif TEST and stations:
        print('  e.g.', {k: stations[0].get(k) for k in ('lat', 'lon', 'alt', 'tC', 'rh', 'p')})
    return 0

# ---- Your spot vs the airport, from a year of NOAA's hourly 2.5 km maps (URMA, already in Modal storage for the Map AI).
# URMA blends every station NOAA has (airports, state networks, more) with terrain, so its square at your place carries
# how your spot runs vs the airport's square: by hour of the day, and on calm clear nights vs breezy ones. (It smooths
# some of the strongest calm-night cold pooling, so night differences may come out a bit small; a home station sharpens it.)
def eastern_hour(t):   # local hour in US Eastern time (EST/EDT by the US rule) for a naive UTC datetime
    y = t.year
    s = dt.datetime(y, 3, 8 + (6 - dt.date(y, 3, 8).weekday()) % 7, 7); e = dt.datetime(y, 11, 1 + (6 - dt.date(y, 11, 1).weekday()) % 7, 6)
    return (t - dt.timedelta(hours=4 if s <= t < e else 5)).hour

def spot_urma():
    data, stp = os.environ.get('SW_GA_DATA', '/data/ga_data'), os.environ.get('SW_STATIONS') or '/data/stations.npz'
    if not os.path.exists(f'{data}/static.npz'): print('No map history in Modal storage.'); return 0
    st = np.load(f'{data}/static.npz'); LA, LO, HG = st['lat'], st['lon'], st['hgt']
    S = np.load(stp) if os.path.exists(stp) else None
    H = {'X-StormWatch-Key': KEY}
    if not TEST and (not API or not KEY): print('Missing the STORMWATCH_API / STORMWATCH_KEY secrets.'); return 1
    places = [(33.81, -84.17)] if TEST else [(p['lat'], p['lon']) for p in json.loads(get(API + '/api/places', headers=H) or b'{}').get('places', [])]
    cosl = math.cos(math.radians(33.8))
    def cell(la, lo):
        d = (LA - la) ** 2 + ((LO - lo) * cosl) ** 2; r, c = np.unravel_index(int(np.argmin(d)), d.shape)
        return int(r), int(c), math.sqrt(float(d[r, c])) * 111.2
    P = []
    for la, lo in places:
        r, c, km = cell(la, lo)
        if km > 3: continue   # off the map
        if S is None: continue
        dk = [math.hypot(int(S['r'][k]) - r, int(S['c'][k]) - c) * 2.5 for k in range(len(S['sid']))]; k = int(np.argmin(dk))
        P.append({'lat': la, 'lon': lo, 'r': r, 'c': c, 'sid': str(S['sid'][k]), 'ar': int(S['r'][k]), 'ac': int(S['c'][k]), 'km': round(dk[k], 1), 'rows': []})
    if not P: print('No places on the map.'); return 0
    files = sorted(f for f in os.listdir(data) if f[:4].isdigit() and f.endswith('.npz'))[-365:]
    tt = time.time()
    for f in files:
        try: z = np.load(f'{data}/{f}'); x, ok = z['x'], z['ok']
        except Exception: continue
        d0 = dt.datetime.fromisoformat(f[:10])
        for p in P:
            for h in range(24):
                if not ok[h]: continue
                th, ta = float(x[h, 0, p['r'], p['c']]), float(x[h, 0, p['ar'], p['ac']])
                dh, da = float(x[h, 1, p['r'], p['c']]), float(x[h, 1, p['ar'], p['ac']])
                u, v, cl = float(x[h, 2, p['ar'], p['ac']]), float(x[h, 3, p['ar'], p['ac']]), float(x[h, 5, p['ar'], p['ac']])
                if not all(map(math.isfinite, (th, ta, u, v))): continue
                p['rows'].append((eastern_hour(d0 + dt.timedelta(hours=h)), (th - ta) * 1.8, (dh - da) * 1.8 if math.isfinite(dh) and math.isfinite(da) else None,
                                  math.hypot(u, v) * 2.237, cl if math.isfinite(cl) else 50.0))
    print(f'URMA history: {len(files)} days read in {time.time() - tt:.0f} s', flush=True)
    med = lambda a: (round(float(np.median(a)), 1), len(a)) if len(a) else (None, 0)
    out = []
    for p in P:
        R = p['rows']; night = lambda h: h >= 20 or h <= 7
        grp = {'day': [r for r in R if not night(r[0])], 'calm_clear_night': [r for r in R if night(r[0]) and r[3] < 4 and r[4] < 30],
               'breezy_or_cloudy_night': [r for r in R if night(r[0]) and not (r[3] < 4 and r[4] < 30)]}
        res = {'lat': p['lat'], 'lon': p['lon'], 'station': p['sid'], 'station_km': p['km'], 'elev_ft': round(float(HG[p['r'], p['c']]) * 3.281), 'station_elev_ft': round(float(HG[p['ar'], p['ac']]) * 3.281),
               'hours': len(R), 'days': len(files)}
        for g, rows in grp.items():
            t, n = med([r[1] for r in rows]); dd, _ = med([r[2] for r in rows if r[2] is not None]); res[g] = {'t': t, 'td': dd, 'n': n}
        res['by_hour'] = [med([r[1] for r in R if r[0] == h])[0] for h in range(24)]
        out.append(res)
        print(f"{p['lat']:.2f},{p['lon']:.2f} vs {p['sid']} ({p['km']} km): your square {res['elev_ft']} ft, airport's {res['station_elev_ft']} ft | day {res['day']} | calm clear night {res['calm_clear_night']} | breezy/cloudy night {res['breezy_or_cloudy_night']}", flush=True)
        print('   by local hour (your spot minus airport, F): ' + ' '.join(f'{h}:{v}' for h, v in enumerate(res['by_hour'])), flush=True)
    if TEST: return 0
    req = urllib.request.Request(API + '/api/spot-urma', data=json.dumps({'spots': out}).encode(), method='POST', headers=dict(UA, **H, **{'Content-Type': 'application/json'}))
    with urllib.request.urlopen(req, timeout=60) as r: print('Sent to your server:', r.read()[:200])
    return 0

if __name__ == '__main__':
    sys.exit(rrfs_long() if '--rrfs-long' in sys.argv else aigefs_long() if '--aigefs' in sys.argv else crowd_snapshot() if '--crowd' in sys.argv else spot_urma() if '--spot' in sys.argv else main())
