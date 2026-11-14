"""Extract the hourly demand snapshot used in Chapter 28 from EIA-930."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

import numpy as np


SOURCE_URL = (
    "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/"
    "EIA930_BALANCE_2026_Jul_Dec.csv"
)
SNAPSHOT = Path(__file__).with_name("data") / "eia930_demand_2026.csv"
AREAS = {"NYIS": "America/New_York", "ERCO": "America/Chicago"}
HOUR = timedelta(hours=1)
START = datetime(2026, 9, 18, tzinfo=timezone.utc)
END = datetime(2026, 11, 13, tzinfo=timezone.utc)
STAMP = "%Y-%m-%dT%H:%M:%SZ"


def timestamps():
    return tuple(START + i * HOUR for i in range((END - START) // HOUR + 1))


def extract_demand(source, destination):
    """Select the two reported demand columns over a fixed UTC window."""
    by_area = {area: {} for area in AREAS}
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            area = row["Balancing Authority"]
            if area not in by_area:
                continue
            stamp = datetime.strptime(row["UTC Time at End of Hour"], "%m/%d/%Y %I:%M:%S %p")
            stamp = stamp.replace(tzinfo=timezone.utc)
            if not START <= stamp <= END:
                continue
            if stamp in by_area[area]:
                raise ValueError(f"Duplicate hour for {area} at {stamp}")
            demand = int(row["Demand (MW)"] or "0")
            if demand <= 0 or row["Demand (MW) (Imputed)"]:
                raise ValueError(f"Expected reported demand for {area} at {stamp}")
            by_area[area][stamp] = demand

    hours = timestamps()
    if any(set(series) != set(hours) for series in by_area.values()):
        raise ValueError("The source must contain every selected hour for both balancing authorities")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["hour_ending_utc", *[f"{area}_mw" for area in AREAS]])
        writer.writerows([stamp.strftime(STAMP), *[by_area[area][stamp] for area in AREAS]] for stamp in hours)


def load_snapshot(path):
    """Return verified hour-ending times and demand arrays, in megawatts."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_fields = ["hour_ending_utc", *[f"{area}_mw" for area in AREAS]]
        if reader.fieldnames != expected_fields:
            raise ValueError("Unexpected demand columns")
        rows = list(reader)
    hours = timestamps()
    if [row["hour_ending_utc"] for row in rows] != [stamp.strftime(STAMP) for stamp in hours]:
        raise ValueError("Expected contiguous hour-ending UTC timestamps")
    demand = {area: np.array([int(row[f"{area}_mw"]) for row in rows]) for area in AREAS}
    if any(np.any(values <= 0) for values in demand.values()):
        raise ValueError("Demand must be positive for this snapshot")
    return hours, demand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="path to the EIA balance CSV")
    parser.add_argument("--fetch", action="store_true", help="download the EIA balance CSV to --raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fetch:
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE_URL, timeout=60) as response:
            args.raw.write_bytes(response.read())
    if args.raw.resolve() == args.output.resolve():
        raise ValueError("Use a separate path for the extracted CSV")
    extract_demand(args.raw, args.output)
    print(f"Source SHA-256  {sha256(args.raw.read_bytes()).hexdigest()}")
    print(f"Snapshot SHA-256  {sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
