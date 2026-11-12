"""
StormWatch Precip Type: what's falling (rain, drizzle, snow, sleet, freezing rain, freezing drizzle, or a mix), read from
the whole temperature profile from the ground up. This file holds the parts shared by training (ptype.py, in this repo)
and the hourly live run (stormwatch-georgia-now run.py, which keeps a byte-for-byte copy: edit both).

  - fetch_points(): the HRRR's temperature + humidity at 17 pressure levels and its surface fields at a list of spots
  - features():     the profile turned into numbers a model can use (Bourgouin-style melting / refreezing energy,
                    warm nose, cold layer near the ground, cloud-top temperature, wet-bulb at the ground ...)
  - Model:          the trained model, saved as plain numpy arrays (no scikit-learn needed to run it, and no
                    version mismatch between the training and hourly images)
  - snow_kind():    wet or dry snow and a rough snow ratio (a physics rule; nothing to train it on yet)

Why the profile: snow falling through a layer above freezing melts; if it then falls through a deep enough cold layer near
the ground it refreezes into sleet, otherwise it hits the ground as freezing rain (the Georgia wedge). The energy of each
layer (Bourgouin 2000: Cp x mean temperature x ln(p_bottom / p_top), in J/kg) says how much melting or refreezing happens.
Models often run 1-2 degrees warm at the ground in a wedge, so the ground's wet-bulb from StormWatch's own surface
forecast (the Map AI, which starts from real observations) is fed in as a separate final check.
"""
import math
import numpy as np

CLASSES = ['rain', 'drizzle', 'snow', 'sleet', 'fzra', 'fzdz', 'mix']
NAMES = {'rain': 'Rain', 'drizzle': 'Drizzle', 'snow': 'Snow', 'sleet': 'Sleet', 'fzra': 'Freezing rain',
         'fzdz': 'Freezing drizzle', 'mix': 'Wintry mix'}
WINTRY = [2, 3, 4, 5, 6]   # everything but rain and drizzle
LEVELS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 600, 550, 500]   # hPa
SFC = [('PRES', 'surface'), ('TMP', '2 m above ground'), ('DPT', '2 m above ground'), ('PRATE', 'surface'),
       ('CPOFP', 'surface'), ('CRAIN', 'surface'), ('CSNOW', 'surface'), ('CICEP', 'surface'), ('CFRZR', 'surface')]
WANT = [('TMP', f'{p} mb') for p in LEVELS] + [('RH', f'{p} mb') for p in LEVELS] + SFC
NL, NW = len(LEVELS), len(WANT)
I_PS, I_T2, I_TD2, I_PR, I_CPOFP, I_CR, I_CS, I_CI, I_CF = range(2 * NL, NW)
HRRR = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com'
CP = 1004.0

# ── HRRR at spots ──
def hrrr_url(init, fh): return f'{HRRR}/hrrr.{init:%Y%m%d}/conus/hrrr.t{init:%H}z.wrfprsf{fh:02d}.grib2'

def ranges(idx_txt):
    """Byte range of each wanted field in one HRRR pressure-level file (None if any is missing)."""
    lines = [l.split(':') for l in idx_txt.strip().splitlines()]
    out = {}
    for i, c in enumerate(lines):
        w = (c[3], c[4]) if len(c) > 4 else None
        if w in WANT and w not in out:
            out[w] = (int(c[1]), int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else '')
    return out if len(out) == NW else None

_grid = {}
def grid_index(msg, lat, lon):
    """Nearest HRRR grid point to each spot (the grid's lat/lon worked out once per process)."""
    if 'tree' not in _grid:
        from scipy.spatial import cKDTree
        la, lo = msg.latlons(); lo = ((lo + 180) % 360) - 180
        m = (la > 24) & (la < 42) & (lo > -95) & (lo < -72)   # the Southeast and a margin: a small search
        rr, cc = np.nonzero(m)
        _grid.update(tree=cKDTree(np.c_[la[rr, cc], lo[rr, cc] * math.cos(math.radians(33))]), rr=rr, cc=cc)
    _, j = _grid['tree'].query(np.c_[lat, ((np.asarray(lon) + 180) % 360 - 180) * math.cos(math.radians(33))])
    return _grid['rr'][j], _grid['cc'][j]

def fetch_points(init, fh, lat, lon, get, workers=6):
    """The HRRR run `init` at forecast hour `fh` at the spots: (N, NW) float32 in GRIB units, or None if not out."""
    import pygrib
    from concurrent.futures import ThreadPoolExecutor
    url = hrrr_url(init, fh); idx = get(url + '.idx')
    if not idx: return None
    R = ranges(idx.decode() if isinstance(idx, bytes) else idx)
    if R is None: return None
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    def one(w):
        b = get(url, R[w])
        if not b: return None
        g = pygrib.fromstring(b); v = g.values
        v = np.ma.filled(v.astype(np.float32), np.nan) if np.ma.isMaskedArray(v) else v.astype(np.float32)
        r, c = grid_index(g, lat, lon)
        return v[r, c]
    with ThreadPoolExecutor(workers) as ex: got = list(ex.map(one, WANT))
    if any(g is None for g in got): return None
    return np.stack(got, 1).astype(np.float32)

def precip_near(init, fh, lat, lon, get, cells=8):
    """The HRRR's heaviest precip rate (mm/h) within about 25 km of each spot (cells x 3 km around it), or None.
    The live run uses it to skip hours with nothing falling nearby (most hours: the full profile is ~30 MB per hour ahead)."""
    import pygrib
    url = hrrr_url(init, fh); idx = get(url + '.idx')
    if not idx: return None
    lines = [l.split(':') for l in (idx.decode() if isinstance(idx, bytes) else idx).strip().splitlines()]
    for i, c in enumerate(lines):
        if len(c) > 4 and c[3] == 'PRATE' and c[4] == 'surface':
            b = get(url, (int(c[1]), int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ''))
            if not b: return None
            g = pygrib.fromstring(b); v = np.ma.filled(g.values.astype(np.float32), 0) if np.ma.isMaskedArray(g.values) else g.values.astype(np.float32)
            r, cc = grid_index(g, np.asarray(lat, float), np.asarray(lon, float))
            return np.array([float(v[max(0, a - cells):a + cells + 1, max(0, d - cells):d + cells + 1].max()) * 3600 for a, d in zip(r, cc)])
    return None

# ── Physics ──
def esat(tc): return 6.112 * np.exp(17.67 * tc / (tc + 243.5))   # hPa over water

def dewpoint(tc, rh):
    e = esat(tc) * np.clip(rh, 1, 100) / 100.0; a = np.log(e / 6.112)
    return 243.5 * a / (17.67 - a)

def wetbulb(tc, tdc, p):
    """Wet-bulb (C) from temperature, dew point (C) and pressure (hPa): the psychrometric equation, solved by bisection."""
    tc, tdc, p = np.broadcast_arrays(np.asarray(tc, float), np.asarray(tdc, float), np.asarray(p, float))
    e = esat(tdc); lo, hi = np.minimum(tdc, tc) - 0.01, tc + 0.01
    for _ in range(30):
        m = (lo + hi) / 2
        f = esat(m) - 6.6e-4 * (1 + 0.00115 * m) * p * (tc - m) - e
        hi = np.where(f > 0, m, hi); lo = np.where(f > 0, lo, m)
    return (lo + hi) / 2

def layers(p, t):
    """Split a profile (ground first, pressure falling) into layers above / below 0 C.
    Returns [(sign, area J/kg, p_bottom, p_top, extreme T)], ground first. Crossings are found on a straight line in ln p."""
    out = []
    cur = None
    for i in range(len(p) - 1):
        p0, p1, t0, t1 = p[i], p[i + 1], t[i], t[i + 1]
        segs = []
        if t0 * t1 < 0:
            f = t0 / (t0 - t1); px = math.exp(math.log(p0) + f * (math.log(p1) - math.log(p0)))
            segs = [(p0, px, t0, 0.0), (px, p1, 0.0, t1)]
        else: segs = [(p0, p1, t0, t1)]
        for a, b, ta, tb in segs:
            s = 1 if (ta + tb) > 0 else -1
            area = CP * (ta + tb) / 2 * math.log(a / b)
            ext = max(ta, tb) if s > 0 else min(ta, tb)
            if cur and cur[0] == s: cur = (s, cur[1] + area, cur[2], b, max(cur[4], ext) if s > 0 else min(cur[4], ext))
            else:
                if cur: out.append(cur)
                cur = (s, area, a, b, ext)
    if cur: out.append(cur)
    return out

def profile(raw):
    """One spot's raw HRRR row -> (p, T, Td) from the ground up (C, hPa), levels below ground dropped."""
    ps = raw[I_PS] / 100.0; t2, td2 = raw[I_T2] - 273.15, raw[I_TD2] - 273.15
    P, T, D = [ps], [t2], [min(td2, t2)]
    for k, lv in enumerate(LEVELS):
        if lv < ps - 5 and np.isfinite(raw[k]):
            tc = raw[k] - 273.15; P.append(lv); T.append(tc); D.append(dewpoint(tc, raw[NL + k]))
    return np.array(P), np.array(T), np.array(D)

def bourgouin(L):
    """Bourgouin (2000) precip type from the layers: the textbook call, kept as a feature and a second baseline."""
    if not L: return 0
    s0 = L[0]
    if s0[0] > 0:   # warm at the ground
        pa = s0[1]
        if len(L) == 1: return 0 if pa > 13.2 else (6 if pa > 5.6 else 2)
        return 0 if pa > 5.6 else (6 if pa > 2 else 2)
    na = -s0[1]   # cold at the ground; is there a warm layer above it?
    warm = [x for x in L[1:] if x[0] > 0]
    if not warm: return 2
    pa = max(x[1] for x in warm)
    if pa < 2: return 2
    if na > 66 + 0.66 * pa: return 3
    if na < 46 + 0.66 * pa: return 4
    return 6

FEATURES = (['lead', 'ps', 't2', 'td2', 'tw2', 'ot2', 'otd2', 'otw2', 'dtw2', 'prate', 'cpofp', 'crain', 'csnow', 'cicep', 'cfrzr',
             'bourgouin', 'nlay']
            + [f'{k}_{v}' for k in ('t', 'w') for v in ('sfc_sign', 'sfc_area', 'sfc_depth', 'warm_area', 'warm_tot', 'warm_max',
                                                       'warm_depth', 'warm_base', 'cold_above', 'tmax_low')]
            + ['tmin_150', 't925', 't850', 't700', 'rh_low', 'rh_mid', 'cloud_top_t', 'sat_dgz', 'ctop_p']
            + [f't_agl{k}' for k in range(8)] + [f'w_agl{k}' for k in range(8)])
NF = len(FEATURES)

def features(raw, ot2=None, otd2=None, lead=1):
    """Raw HRRR rows (N, NW) + the ground's temperature / dew point from observations or StormWatch's surface forecast
    (C; None = use the HRRR's own) -> (N, NF) features, and the Bourgouin call for each row."""
    raw = np.atleast_2d(raw); N = len(raw)
    X = np.full((N, NF), np.nan, np.float32); B = np.zeros(N, int)
    ot2 = np.full(N, np.nan) if ot2 is None else np.broadcast_to(np.asarray(ot2, float), (N,))
    otd2 = np.full(N, np.nan) if otd2 is None else np.broadcast_to(np.asarray(otd2, float), (N,))
    lead = np.broadcast_to(np.asarray(lead, float), (N,))
    for i in range(N):
        r = raw[i]
        if not np.isfinite(r[I_PS]) or not np.isfinite(r[I_T2]): continue
        P, T, D = profile(r); W = wetbulb(T, D, P)
        ps, t2, td2 = P[0], T[0], D[0]; tw2 = W[0]
        o_t, o_td = (ot2[i], otd2[i]) if np.isfinite(ot2[i]) and np.isfinite(otd2[i]) else (t2, td2)
        o_td = min(o_td, o_t); otw = float(wetbulb(o_t, o_td, ps))
        f = {'lead': lead[i], 'ps': ps, 't2': t2, 'td2': td2, 'tw2': tw2, 'ot2': o_t, 'otd2': o_td, 'otw2': otw, 'dtw2': otw - tw2,
             'prate': r[I_PR] * 3600, 'cpofp': r[I_CPOFP], 'crain': r[I_CR], 'csnow': r[I_CS], 'cicep': r[I_CI], 'cfrzr': r[I_CF]}
        # the ground's numbers from observations / the surface forecast replace the HRRR's at the bottom of the profile:
        # the "final check" for the wedge (the HRRR often runs warm at the ground there)
        T2, W2 = T.copy(), W.copy(); T2[0], W2[0] = o_t, otw
        for key, prof in (('t', T2), ('w', W2)):
            L = layers(P, prof)
            if key == 't': B[i] = bourgouin(L); f['bourgouin'] = B[i]; f['nlay'] = len(L)
            s0 = L[0] if L else (1, 0, ps, ps, prof[0])
            warm = [x for x in L[1:] if x[0] > 0]
            big = max(warm, key=lambda x: x[1]) if warm else None
            f[f'{key}_sfc_sign'] = s0[0]; f[f'{key}_sfc_area'] = abs(s0[1]); f[f'{key}_sfc_depth'] = s0[2] - s0[3]
            f[f'{key}_warm_area'] = big[1] if big else 0.0; f[f'{key}_warm_tot'] = sum(x[1] for x in warm)
            f[f'{key}_warm_max'] = big[4] if big else np.nan; f[f'{key}_warm_depth'] = (big[2] - big[3]) if big else 0.0
            f[f'{key}_warm_base'] = (ps - big[2]) if big else np.nan
            f[f'{key}_cold_above'] = -sum(x[1] for x in L[1:] if x[0] < 0 and big and x[2] <= big[3])
            low = P >= ps - 300; f[f'{key}_tmax_low'] = float(np.max(prof[low]))
        f['tmin_150'] = float(np.min(T2[P >= ps - 150]))
        for lv in (925, 850, 700):
            k = LEVELS.index(lv); f[f't{lv}'] = r[k] - 273.15
        rh = 100 * esat(D) / esat(T)
        f['rh_low'] = float(np.mean(rh[P >= ps - 200])); f['rh_mid'] = float(np.mean(rh[(P < ps - 200) & (P >= 500)])) if np.any((P < ps - 200) & (P >= 500)) else np.nan
        # cloud top: the top of the lowest deep moist layer. Warmer than about -10 C there's little ice to start snow, so
        # what falls is drizzle (or freezing drizzle on a cold ground): shallow wedge clouds do this a lot
        sat = rh >= 85; top = None
        if sat.any():
            k0 = int(np.argmax(sat)); k = k0
            while k + 1 < len(P) and sat[k + 1]: k += 1
            top = k
        f['cloud_top_t'] = float(T[top]) if top is not None else np.nan; f['ctop_p'] = float(P[top]) if top is not None else np.nan
        dgz = (T <= -10) & (T >= -20)
        f['sat_dgz'] = float(np.max(rh[dgz])) if dgz.any() else np.nan
        # the bottom 8 steps of the profile at fixed heights above ground (every 25 hPa from the ground), T and wet-bulb
        lp = np.log(P)
        for k in range(8):
            q = math.log(ps - 25 * k) if ps - 25 * k > P[-1] else None
            f[f't_agl{k}'] = float(np.interp(-q, -lp, T2)) if q is not None else np.nan
            f[f'w_agl{k}'] = float(np.interp(-q, -lp, W2)) if q is not None else np.nan
        X[i] = [f.get(n, np.nan) for n in FEATURES]
    return X, B

def hrrr_call(raw):
    """The HRRR's own precip type (its yes/no flags) as one of CLASSES: the baseline to beat. HRRR has no drizzle classes."""
    raw = np.atleast_2d(raw); out = np.zeros(len(raw), int)
    for i, r in enumerate(raw):
        on = [c for c, k in ((0, I_CR), (2, I_CS), (3, I_CI), (4, I_CF)) if r[k] > 0.5]
        if len(on) > 1: out[i] = 6
        elif on: out[i] = on[0]
        else:   # no precip in the HRRR here: its percent-frozen, else the ground's wet-bulb
            cp, tw = r[I_CPOFP], float(wetbulb(r[I_T2] - 273.15, min(r[I_TD2], r[I_T2]) - 273.15, r[I_PS] / 100))
            out[i] = 2 if cp >= 60 else (6 if cp >= 30 else (4 if r[I_T2] < 273.15 else (2 if tw <= 0.5 else 0)))
    return out

def snow_kind(x):
    """Wet or dry snow and a rough snow ratio, from one row of features (a physics rule, not trained).
    The ratio follows Kuchera: colder air through the column -> fluffier snow."""
    g = dict(zip(FEATURES, x)); tw, tmax = g['otw2'], g['t_tmax_low']
    ratio = 12 + 2 * (-2 - tmax) if tmax > -2 else 12 + (-2 - tmax)
    melt = g['t_sfc_area'] if g['t_sfc_sign'] > 0 else 0.0
    if tw > -0.5 or melt > 2: kind, ratio = 'wet', min(ratio, 9)
    elif tw < -4 and tmax < -5: kind = 'dry'
    else: kind = 'normal'
    return kind, int(round(min(25, max(5, ratio))))

# ── The trained model, as plain arrays ──
def export(clf, path, extra=None):
    """A fitted HistGradientBoostingClassifier -> .npz any numpy can run (see Model)."""
    nodes, start, cls = [], [], []
    for it in clf._predictors:
        for k, tr in enumerate(it):
            start.append(sum(len(n) for n in nodes)); cls.append(k)
            n = tr.nodes; nodes.append(np.stack([n['value'], n['feature_idx'], n['num_threshold'], n['missing_go_to_left'],
                                                 n['left'], n['right'], n['is_leaf']], 1))
    np.savez_compressed(path, nodes=np.concatenate(nodes).astype(np.float64), start=np.array(start), cls=np.array(cls),
                        base=np.asarray(clf._baseline_prediction, np.float64).ravel(), classes=np.asarray(clf.classes_),
                        **(extra or {}))

class Model:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.nodes, self.start, self.cls, self.base = z['nodes'], z['start'], z['cls'], z['base']
        self.classes = z['classes']; self.prior = z['prior'] if 'prior' in z else np.ones(len(self.classes))
        self.meta = {k: z[k] for k in z.files if k not in ('nodes', 'start', 'cls', 'base', 'classes', 'prior')}
    def raw(self, X):
        X = np.asarray(X, np.float64); out = np.tile(self.base, (len(X), 1))
        nd = self.nodes
        for s, k in zip(self.start, self.cls):
            j = np.full(len(X), s, np.int64)
            for _ in range(64):
                leaf = nd[j, 6] > 0.5
                if leaf.all(): break
                f = nd[j, 1].astype(int); x = X[np.arange(len(X)), f]
                left = np.where(np.isnan(x), nd[j, 3] > 0.5, x <= nd[j, 2])
                nxt = s + np.where(left, nd[j, 4], nd[j, 5]).astype(np.int64)
                j = np.where(leaf, j, nxt)
            out[:, k] += nd[j, 0]
        return out
    def proba(self, X, calibrated=True):
        """Chances for each class. calibrated=True undoes the extra weight rare classes got in training, so the numbers
        are real chances; False keeps it (the type call: leans toward catching the rare mixes)."""
        r = self.raw(X); r -= r.max(1, keepdims=True); p = np.exp(r); p /= p.sum(1, keepdims=True)
        if calibrated: p = p / self.prior; p /= p.sum(1, keepdims=True)
        return p

# ── The live run (called by stormwatch-georgia-now run.py inside the hourly job) ──
COLD_F = 40.0   # only hours ahead where a place is at or below 40 F get a precip type (the rest of the year: nothing to do)

def live(model_path, t0, spots, get, log=print):
    """spots: [(lat, lon, leads)] with leads = StormWatch's surface forecast [{valid, lead, t, td}] (F) for hours 1-18 after
    t0 (the map's hour). Returns, for each spot, [{valid, lead, type, p, wintry, hrrr, snow?, ratio?}] for the cold hours
    with precip near it in the HRRR (empty when it's warm or dry everywhere)."""
    import datetime as dt
    out = [[] for _ in spots]
    if not spots: return out
    cold = sorted({x['lead'] for _, _, L in spots for x in L if x.get('t') is not None and x['t'] <= COLD_F and 1 <= x['lead'] <= 18})
    if not cold: log('Precip type: no place at 40 F or colder in the next 18 hours; skipped.'); return out
    M = Model(model_path); lat = [s[0] for s in spots]; lon = [s[1] for s in spots]
    init = None
    for back in (0, 1, 2, 3):   # the newest HRRR run that reaches the furthest cold hour
        i0 = t0 - dt.timedelta(hours=back)
        if max(cold) + back <= 18 and get(hrrr_url(i0, max(cold) + back) + '.idx'): init, B = i0, back; break
    if init is None: log('Precip type: no HRRR run out yet that covers the cold hours; skipped.'); return out
    wet = []
    for k in cold:
        pr = precip_near(init, k + B, lat, lon, get)
        if pr is not None and (pr > 0.02).any(): wet.append(k)
    use = sorted({j for k in wet for j in (k - 1, k, k + 1) if j in cold})   # and the hour on each side (timing errors)
    if not use: log(f'Precip type: cold hours {cold[0]}-{cold[-1]} ahead, but nothing falling nearby in the HRRR ({init:%H}z); skipped.'); return out
    for k in use:
        raw = fetch_points(init, k + B, lat, lon, get)
        if raw is None: continue
        sf = [next((x for x in L if x['lead'] == k), None) for _, _, L in spots]
        ot = [((x['t'] - 32) / 1.8) if x and x.get('t') is not None else np.nan for x in sf]
        otd = [((x['td'] - 32) / 1.8) if x and x.get('td') is not None else np.nan for x in sf]
        X, _ = features(raw, ot, otd, k); P, W, H = M.proba(X), M.proba(X, calibrated=False), hrrr_call(raw)
        for i, x in enumerate(sf):
            if x is None or x.get('t') is None or x['t'] > COLD_F: continue
            c = int(W[i].argmax()); h = {'valid': x['valid'], 'lead': k, 'type': CLASSES[c],
                                         'p': {CLASSES[j]: int(round(100 * P[i, j])) for j in range(len(CLASSES))},
                                         'wintry': int(round(100 * P[i, WINTRY].sum())), 'hrrr': CLASSES[int(H[i])]}
            if c == 2 or P[i, 2] >= 0.3: h['snow'], h['ratio'] = snow_kind(X[i])
            out[i].append(h)
    n = sum(len(o) for o in out)
    log(f'Precip type: {len(use)} hour(s) ahead with cold air and precip nearby (HRRR {init:%H}z): {n} place-hours'
        + (', e.g. ' + ', '.join(f'{h["lead"]}h {NAMES[h["type"]]} (wintry {h["wintry"]}%)' for h in next(o for o in out if o)[:4]) if n else ''))
    return out
