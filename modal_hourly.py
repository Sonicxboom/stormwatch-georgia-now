"""
Georgia Now on Modal: the same hourly run as GitHub's, on Modal's clock (GitHub's schedule can skip or run late).
Modal starts run.py at :55 every hour; GitHub's hourly run is now the backup and only sends when Modal didn't.

Set up by the "Deploy to Modal" workflow (Actions -> Deploy to Modal -> Run workflow), which copies this repo's
STORMWATCH_API / STORMWATCH_KEY secrets into a Modal secret named "stormwatch-server" and deploys this file.
About a minute of CPU an hour: well under $1 a month of Modal's free credits.
"""
import os, sys, subprocess
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App('stormwatch-hourly')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy<2.3', 'pygrib==2.1.6')
         .pip_install('torch==2.5.1', index_url='https://download.pytorch.org/whl/cpu')
         .add_local_file(os.path.join(HERE, 'run.py'), '/root/gn/run.py')
         .add_local_dir(os.path.join(HERE, 'model'), '/root/gn/model'))

@app.function(image=image, cpu=2, memory=4096, timeout=15 * 60, retries=1,
              secrets=[modal.Secret.from_name('stormwatch-server')], schedule=modal.Cron('55 * * * *'))
def hourly():
    r = subprocess.run([sys.executable, '-u', '/root/gn/run.py'], cwd='/root/gn')
    if r.returncode: raise RuntimeError(f'run.py failed (exit {r.returncode})')

@app.local_entrypoint()
def main():
    hourly.remote()   # `modal run modal_hourly.py`: one run now, to check it works
