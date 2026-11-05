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
         .pip_install('numpy<2.3', 'pygrib==2.1.6')
         .pip_install('torch==2.5.1', index_url='https://download.pytorch.org/whl/cpu')
         .add_local_file(os.path.join(HERE, 'run.py'), '/root/gn/run.py')
         .add_local_dir(os.path.join(HERE, 'model'), '/root/gn/model'))
LIVE, LIVE_PLUS, REGRID = '/data/live/stormwatch_now.pt', '/data/live/stormwatch_now_plus.pt', '/data/hrrr_data/regrid.npz'

def go(*args):
    vol.reload()
    on = lambda f: f if os.path.exists(f) else ''
    env = dict(os.environ, SW_NOW_MODEL=on(LIVE), SW_NOW_PLUS_MODEL=on(LIVE_PLUS), SW_HRRR_REGRID=on(REGRID))
    r = subprocess.run([sys.executable, '-u', '/root/gn/run.py', *args], cwd='/root/gn', env=env)
    if r.returncode: raise RuntimeError(f'run.py failed (exit {r.returncode})')

@app.function(image=image, cpu=2, memory=4096, timeout=15 * 60, retries=1, volumes={'/data': vol},
              secrets=[modal.Secret.from_name('stormwatch-server')], schedule=modal.Cron('55 * * * *'))
def hourly():
    go()

@app.function(image=image, cpu=2, memory=4096, timeout=15 * 60, volumes={'/data': vol})
def check():
    go('--test')   # a forecast for Stone Mountain, printed; nothing is sent

@app.local_entrypoint()
def main():
    check.remote()   # `modal run modal_hourly.py`: the test run
