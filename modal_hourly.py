"""
Georgia Now on Modal: the same hourly run as GitHub's, on Modal's clock (GitHub's schedule can skip or run late).
Modal starts run.py at :55 every hour; GitHub's hourly run is now the backup and only sends when Modal didn't.

The forecast model: StormWatchNow when one has been promoted into Modal storage (live/stormwatch_now.pt in the
"stormwatch-data" volume: Actions -> Train on Modal -> promote, in the StormWatch repo), plus StormWatchNow+ from 3 hours
ahead when one is promoted too (live/stormwatch_now_plus.pt; it also reads the HRRR's newest run); otherwise Georgia Now.

Set up by the "Deploy to Modal" workflow (Actions -> Deploy to Modal -> Run workflow), which copies this repo's
STORMWATCH_API / STORMWATCH_KEY secrets into a Modal secret named "stormwatch-server", deploys this file, then does a
test run that prints a sample forecast and sends nothing. About a minute of CPU an hour: well under $1-3 a month.
"""
import os, sys, subprocess
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App('stormwatch-hourly')
vol = modal.Volume.from_name('stormwatch-data', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy<2.3', 'pygrib==2.1.6', 'scipy', 'joblib', 'threadpoolctl', 'scikit-learn')   # sklearn: StormWatch Long
         .pip_install('torch==2.5.1', index_url='https://download.pytorch.org/whl/cpu')
         .add_local_file(os.path.join(HERE, 'run.py'), '/root/gn/run.py')
         .add_local_dir(os.path.join(HERE, 'model'), '/root/gn/model'))
LIVE_TEST = '/data/live_test'   # the live airport test: each run's forecasts at ~170 airports, scored every 6 hours
LIVE, LIVE_PLUS, REGRID = '/data/live/stormwatch_now.pt', '/data/live/stormwatch_now_plus.pt', '/data/hrrr_data/regrid.npz'

def go(*args):
    vol.reload()
    on = lambda f: f if os.path.exists(f) else ''
    env = dict(os.environ, SW_NOW_MODEL=on(LIVE), SW_NOW_PLUS_MODEL=on(LIVE_PLUS), SW_HRRR_REGRID=on(REGRID),
               SW_LIVE_TEST=LIVE_TEST, SW_STATIONS=on('/data/stations.npz'),
               SW_RRFS_SAVE='/data/rrfs_data',   # the RRFS's Georgia maps, kept for training the Map AI
               SW_MAPAI_SAVE='/data/mapai_now')  # Map AI's "right now" map each hour: the 3D-Var's starting guess
    r = subprocess.run([sys.executable, '-u', '/root/gn/run.py', *args], cwd='/root/gn', env=env)
    if not args: vol.commit()   # keeps this hour's live-test file
    if r.returncode: raise RuntimeError(f'run.py failed (exit {r.returncode})')

@app.function(image=image, cpu=2, memory=4096, timeout=15 * 60, retries=1, volumes={'/data': vol},
              secrets=[modal.Secret.from_name('stormwatch-server')], schedule=modal.Cron('55 * * * *'))
def hourly():
    go()

# NOAA's RRFS 84-hour runs (00/06/12/18 UTC, finished ~2.5 hours later) at your places, for the 7-day blend and the daily
# scoreboard (it replaced the NAM in Nov 2026). About 2 minutes, 4 times a day.
@app.function(image=image, cpu=2, memory=4096, timeout=20 * 60, retries=1,
              secrets=[modal.Secret.from_name('stormwatch-server')], schedule=modal.Cron('7 3,9,15,21 * * *'))
def rrfs_long():
    r = subprocess.run([sys.executable, '-u', '/root/gn/run.py', '--rrfs-long'], cwd='/root/gn')
    if r.returncode: raise RuntimeError(f'run.py --rrfs-long failed (exit {r.returncode})')

@app.function(image=image, cpu=2, memory=4096, timeout=20 * 60)
def check_long():
    subprocess.run([sys.executable, '-u', '/root/gn/run.py', '--rrfs-long', '--test'], cwd='/root/gn')   # prints Stone Mountain, sends nothing

# NOAA's AI ensemble (AIGEFS), 10 of its 31 members, for the 7-day blend; its Georgia box is kept in /data/aigefs_data to
# train and run StormWatch Long. Its runs land on AWS 6-10 hours after they start, so 4 tries a day take the newest.
@app.function(image=image, cpu=2, memory=4096, timeout=25 * 60, retries=1, volumes={'/data': vol},
              secrets=[modal.Secret.from_name('stormwatch-server')], schedule=modal.Cron('15 3,9,15,21 * * *'))
def aigefs():
    vol.reload()   # StormWatch Long's newest model from Kaggle (live/ or runs/*-long/)
    r = subprocess.run([sys.executable, '-u', '/root/gn/run.py', '--aigefs'], cwd='/root/gn', env=dict(os.environ, SW_AIGEFS_SAVE='/data/aigefs_data', SW_LONG_DIR='/data'))
    vol.commit()
    if r.returncode: raise RuntimeError(f'run.py --aigefs failed (exit {r.returncode})')

# The crowd layer: every public Netatmo home station across Georgia, each hour (kept 60 days in /data/crowd)
@app.function(image=image, cpu=1, memory=1024, timeout=10 * 60, volumes={'/data': vol},
              secrets=[modal.Secret.from_name('stormwatch-server')], schedule=modal.Cron('48 * * * *'))
def crowd():
    subprocess.run([sys.executable, '-u', '/root/gn/run.py', '--crowd'], cwd='/root/gn', env=dict(os.environ, SW_CROWD_SAVE='/data/crowd'))
    vol.commit()
    import datetime as _dt
    now = _dt.datetime.utcnow()
    if now.weekday() == 0 and now.hour == 4:   # once a week (Monday 4 UTC): your spot vs the airport from the URMA maps
        subprocess.run([sys.executable, '-u', '/root/gn/run.py', '--spot'], cwd='/root/gn', env=dict(os.environ, SW_GA_DATA='/data/ga_data', SW_STATIONS='/data/stations.npz'))

@app.function(image=image, cpu=1, memory=1024, timeout=10 * 60, secrets=[modal.Secret.from_name('stormwatch-server')])
def check_crowd():
    subprocess.run([sys.executable, '-u', '/root/gn/run.py', '--crowd', '--test'], cwd='/root/gn')   # counts them, saves nothing

@app.function(image=image, cpu=2, memory=4096, timeout=25 * 60, volumes={'/data': vol})
def check_aigefs():
    subprocess.run([sys.executable, '-u', '/root/gn/run.py', '--aigefs', '--test'], cwd='/root/gn', env=dict(os.environ, SW_LONG_DIR='/data'))   # Stone Mountain, sends nothing

@app.function(image=image, cpu=2, memory=4096, timeout=15 * 60, volumes={'/data': vol})
def check():
    go('--test')   # a forecast for Stone Mountain, printed; nothing is sent

# ---- The live airport test: every hourly run checked at all ~170 airports once its hours have passed ----
# Same airports and same scoring as the airport test on past weeks, but on the real live runs. Read-only: it never
# changes a forecast. Scores add up in live_test/score.json; `modal run modal_hourly.py::report` prints them.
NAMES = {'map_ai': 'Map AI (as sent)', 'now': 'StormWatchNow', 'plus': 'StormWatchNow+', 'hrrr': 'HRRR'}

def airport_obs(sids, a, b):
    """Each airport's routine hourly reading (the :45-:59 one, filed under the next hour) between a and b, from IEM."""
    import urllib.request, datetime as dt
    out = {}
    for k in range(0, len(sids), 40):   # 40 airports per request
        q = '&'.join(f'station={x}' for x in sids[k:k + 40])
        u = (f'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?{q}&data=tmpf&data=dwpf&data=sknt&year1={a.year}&month1={a.month}&day1={a.day}'
             f'&year2={b.year}&month2={b.month}&day2={b.day}&tz=Etc/UTC&format=onlycomma&latlon=no&missing=empty&report_type=3')
        try: txt = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'StormWatch personal weather app (live verification)'}), timeout=120).read().decode()
        except Exception as e: print('  (readings unavailable:', e, ')'); continue
        for ln in txt.splitlines()[1:]:
            p = ln.split(',')
            if len(p) < 5 or not p[2]: continue
            try: tm = dt.datetime.strptime(p[1], '%Y-%m-%d %H:%M')
            except ValueError: continue
            if tm.minute < 45: continue
            h = (tm + dt.timedelta(minutes=15)).replace(minute=0)
            f = lambda v: float(v) if v else None
            out[(p[0], h)] = (f(p[2]), f(p[3]), f(p[4]) * 1.15078 if p[4] else None)
    return out

@app.function(image=image, timeout=30 * 60, volumes={'/data': vol}, schedule=modal.Cron('20 */6 * * *'))
def live_score():
    import json, glob, datetime as dt
    import numpy as np
    vol.reload()
    runs = sorted(glob.glob(f'{LIVE_TEST}/runs/*.npz')); now = dt.datetime.utcnow()
    ready = [f for f in runs if dt.datetime.strptime(os.path.basename(f)[:10], '%Y%m%d%H') + dt.timedelta(hours=20) < now]
    if not ready: print('Nothing ready to score yet (a run is scored once all 18 of its hours have passed).'); return
    Z = [np.load(f, allow_pickle=True) for f in ready]
    t0s = [dt.datetime.fromisoformat(str(z['issued'])) for z in Z]
    sids = sorted({str(x) for z in Z for x in z['sid']})
    obs = airport_obs(sids, min(t0s) - dt.timedelta(days=1), max(t0s) + dt.timedelta(days=2))
    sf = f'{LIVE_TEST}/score.json'
    S = json.load(open(sf)) if os.path.exists(sf) else {'runs': 0, 'since': t0s[0].isoformat(), 'mae': {}}
    for z, t0 in zip(Z, t0s):
        leads, sid = [int(x) for x in z['leads']], [str(x) for x in z['sid']]
        srcs = [k for k in NAMES if k in z.files]
        for li, L in enumerate(leads):
            vt = t0 + dt.timedelta(hours=L)
            for si, s_ in enumerate(sid):
                o = obs.get((s_, vt))
                if not o: continue
                for vi, vn in enumerate(('temp', 'dew', 'wind')):
                    if o[vi] is None: continue
                    vals = {k: float(z[k][li, si, vi]) for k in srcs}
                    if any(v != v for v in vals.values()): continue   # same airports and hours for every source
                    for k, v in vals.items():
                        c = S['mae'].setdefault(k, {}).setdefault(vn, {}).setdefault(str(L), [0.0, 0])
                        c[0] += abs(v - o[vi]); c[1] += 1
        S['runs'] += 1
    os.makedirs(f'{LIVE_TEST}/scored', exist_ok=True)
    for f in ready: os.replace(f, f'{LIVE_TEST}/scored/' + os.path.basename(f))
    S['updated'] = now.isoformat(timespec='minutes'); json.dump(S, open(sf, 'w')); vol.commit()
    print(f'Scored {len(ready)} run(s); {S["runs"]} so far.'); print(table(S))

def table(S):
    out = [f'Live airport test: {S["runs"]} hourly runs since {S["since"][:16]} UTC, mean miss vs what the airports measured']
    for vn, u in (('temp', 'F'), ('dew', 'F'), ('wind', 'mph')):
        leads = sorted({int(L) for k in S['mae'] for L in S['mae'][k].get(vn, {})})
        if not leads: continue
        out.append(f'  {vn} ({u})'.ljust(28) + ''.join(f'{L:>7}' for L in leads))
        for k in NAMES:
            c = S['mae'].get(k, {}).get(vn)
            if c: out.append(f'    {NAMES[k]}'.ljust(28) + ''.join(f'{c[str(L)][0] / c[str(L)][1]:>7.2f}' if str(L) in c and c[str(L)][1] else '      -' for L in leads))
    return '\n'.join(out)

@app.function(image=image, timeout=300, volumes={'/data': vol})
def report():
    import json
    vol.reload(); sf = f'{LIVE_TEST}/score.json'
    n = len(os.listdir(f'{LIVE_TEST}/runs')) if os.path.isdir(f'{LIVE_TEST}/runs') else 0
    msg = table(json.load(open(sf))) if os.path.exists(sf) else 'No live scores yet (the first runs are scored about 20 hours after they run).'
    print(msg + f'\n({n} run(s) waiting to be scored)'); return msg

# Your spot vs the airport from the year of NOAA maps (URMA): sent weekly from the crowd job above (Modal's free plan
# allows 5 scheduled jobs, all in use); this one prints Stone Mountain's numbers in the deploy's test run
@app.function(image=image, cpu=2, memory=4096, timeout=25 * 60, volumes={'/data': vol})
def check_spot():
    subprocess.run([sys.executable, '-u', '/root/gn/run.py', '--spot', '--test'], cwd='/root/gn', env=dict(os.environ, SW_GA_DATA='/data/ga_data', SW_STATIONS='/data/stations.npz'))   # Stone Mountain, printed

@app.local_entrypoint()
def main():
    check.remote()   # `modal run modal_hourly.py`: the test run
    check_long.remote()   # and the RRFS 84-hour run at Stone Mountain (also sends nothing)
    check_aigefs.remote()   # and NOAA's AI ensemble at Stone Mountain (also sends nothing)
    check_crowd.remote()    # and the crowd layer: how many Netatmo stations it finds (saves nothing)
    check_spot.remote()     # and your spot vs the airport from a year of NOAA maps (printed)
