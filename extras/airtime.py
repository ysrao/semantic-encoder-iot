"""Paper v2.0, Section V, Table IV. LoRa time on air (Semtech AN1200.13 / SX1276 datasheet formula), EU868, BW 125 kHz,
CR 4/5, explicit header, CRC on, 8-symbol preamble, low-data-rate optimisation at SF11-12.
LoRaWAN adds 13 B to each application payload (MHDR 1, FHDR 7, FPort 1, MIC 4).
Max application payload EU868 (no FOpts): SF7-8 222 B, SF9 115 B, SF10-12 51 B."""
import math, json
OVH = 13; MAXPL = {7: 222, 8: 222, 9: 115, 10: 51, 11: 51, 12: 51}
def toa_ms(app, sf, bw=125e3):
    pl = app + OVH; de = 1 if sf >= 11 else 0; ts = 2 ** sf / bw
    n = 8 + max(math.ceil((8 * pl - 4 * sf + 28 + 16) / (4 * (sf - 2 * de))) * 5, 0)
    return ((8 + 4.25) + n) * ts * 1000
def msg_ms(app, sf):           # fragment if over the max payload
    k = math.ceil(app / MAXPL[sf]); full, last = divmod(app, MAXPL[sf])
    frames = [MAXPL[sf]] * full + ([last] if last else [])
    return sum(toa_ms(f, sf) for f in frames), len(frames)
I_TX, V = 0.044, 3.3   # assumed: ~44 mA at +14 dBm, 3.3 V supply (SX127x class)
out = {}
for sf in (7, 10, 12):
    row = {}
    for name, b in (("learned 8 B", 8), ("reference 20 B", 20), ("raw 256 B", 256)):
        ms, fr = msg_ms(b, sf); row[name] = {"toa_ms": round(ms, 1), "frames": fr, "energy_mJ": round(ms / 1000 * I_TX * V * 1000, 2)}
    out[sf] = row
print(json.dumps(out, indent=1))
# per day, at the paper's cadences
cad = {"flood (1 window / 8 h)": 3, "outage (1 / 8 s)": 10800, "metering (1 / 8 min)": 180, "parking (1 / min)": 1440}
for sf in (7, 10, 12):
    for k, n in cad.items():
        l, _ = msg_ms(8, sf); r, _ = msg_ms(256, sf)
        print(f"SF{sf} {k}: learned {n*l/1000:.0f} s/day ({n*l/864e3:.2f} % airtime), raw {n*r/1000:.0f} s/day ({n*r/864e3:.2f} %)")
# event-triggered reference (paper's bytes/day): metering 18.5 B/day -> 0.925 reports/day; parking 496 -> 24.8 reports/day
for sf in (7, 12):
    r, _ = msg_ms(20, sf); l, _ = msg_ms(8, sf)
    print(f"SF{sf} metering: reference {0.925*r/1000:.2f} s/day vs learned {180*l/1000:.1f} s/day; parking: reference {24.8*r/1000:.1f} s/day vs learned {1440*l/1000:.0f} s/day")
