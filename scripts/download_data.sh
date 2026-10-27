#!/usr/bin/env bash
# Downloads both datasets into ./data.
set -euo pipefail
DATA_DIR="${DATA_DIR:-$(dirname "$0")/../data}"
mkdir -p "$DATA_DIR"
# Spain load + operator forecast, 2015-2018 (ENTSO-E, via the darts repo)
curl -fsSL -o "$DATA_DIR/energy_dataset.csv" https://raw.githubusercontent.com/unit8co/darts/master/datasets/energy_dataset.csv
# PJM East, 2002-2018
curl -fsSL -o "$DATA_DIR/PJME_hourly.csv" https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJME_hourly.csv
echo "Data saved to $DATA_DIR"
