#!/usr/bin/env python
"""Fetch hourly weather (rain + temperature) for the six CWC stations from
the Open-Meteo archive API (ERA5 reanalysis, free, no key), aligned to IST.

Exactly what is fetched, per station:
  GET https://archive-api.open-meteo.com/v1/archive
      ?latitude=<lat>&longitude=<lon>
      &start_date=2020-01-01&end_date=2024-07-31
      &hourly=temperature_2m,rain
      &timezone=Asia/Kolkata
Cached verbatim to weather_<station>.json in this folder.
"""
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

STATIONS = {
    "Jainagar":    (26.583, 86.148),
    "Jhanjharpur": (26.233, 86.259),
    "Benibad":     (26.145, 85.692),
    "Kallooppara": (9.404, 76.650),
    "Akhuapada":   (20.916, 86.275),
    "Kunavaram":   (17.573, 81.250),
}

URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}"
       "&longitude={lon}&start_date=2020-01-01&end_date=2024-07-31"
       "&hourly=temperature_2m,rain&timezone=Asia%2FKolkata")


def main():
    for name, (lat, lon) in STATIONS.items():
        out = os.path.join(HERE, f"weather_{name}.json")
        if os.path.exists(out):
            print(f"{name}: cached, skip")
            continue
        url = URL.format(lat=lat, lon=lon)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode("utf-8"))
        with open(out, "w") as f:
            json.dump(data, f)
        h = data["hourly"]
        print(f"{name}: {len(h['time'])} hours, vars={list(h.keys())}, "
              f"first={h['time'][0]} last={h['time'][-1]}, "
              f"rain NaN={sum(1 for v in h['rain'] if v is None)}, "
              f"temp NaN={sum(1 for v in h['temperature_2m'] if v is None)}")


if __name__ == "__main__":
    main()
