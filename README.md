# Georgia Now (StormWatch)

A personal weather project. Every hour, NOAA's newest real-time weather maps of Georgia (RTMA, 2.5 km) go into
Georgia Now, a small neural network trained on years of NOAA's hourly maps, which steps them forward 18 hours
(temperature, dew point, wind, gusts, cloud cover). The result goes to a private server.

- `run.py` - the hourly run (`python run.py --test` prints a sample and sends nothing)
- `model/` - the trained network and its map
- `.github/workflows/hourly.yml` - the schedule

Data: NOAA RTMA via the NOAA Open Data Dissemination program on AWS.
