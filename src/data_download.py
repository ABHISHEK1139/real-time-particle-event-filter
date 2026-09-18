"""Fetch the CERN Open Data educational dimuon sample and materialise a ROOT copy.

Dataset identity (fixed): CMS DoubleMu Run2011A, proton-proton collisions at
7 TeV. Educational 100k-event derived sample:
  https://opendata.cern.ch/record/5201
(parent: Datasets derived from the Run2011A SingleElectron/SingleMu/
DoubleElectron/DoubleMu primary datasets, https://opendata.cern.ch/record/545).

NOTE: this is an education/outreach subset, not suitable for a full physics
analysis. Previously the repo mixed this up with the Run2010B sample
(https://opendata.cern.ch/record/700) and cited 13.6 TeV (Run-3 energy).
Both were wrong for this data; the correct energy here is 7 TeV.
"""
import os
import time
import urllib.request
import pandas as pd
import numpy as np
import uproot

# Canonical educational CSV (record 5201). The old record/545 URL is kept as a
# fallback mirror because 5201's parent is 545 and mirrors have shifted before.
DATA_URLS = [
    "https://opendata.cern.ch/record/5201/files/Dimuon_DoubleMu.csv",
    "https://opendata.cern.ch/record/545/files/Dimuon_DoubleMu.csv",
]
CSV_FILE = "Dimuon_DoubleMu.csv"
ROOT_FILE = "Dimuon_DoubleMu.root"

REQUIRED_COLUMNS = {'pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2', 'M'}
# Sanity bounds for the educational sample (M selection is 0.3-300 GeV upstream,
# derived file used here spans ~2-110+ GeV; be permissive, catch corruption).
MIN_ROWS = 1000
MIN_FILE_BYTES = 100_000


def _download_with_retry(dest, urls=DATA_URLS, retries=3, timeout=120):
    last_err = None
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                print(f"📥 Downloading {url} (attempt {attempt}/{retries})...")
                urllib.request.urlretrieve(url, dest)
                return url
            except Exception as e:  # network errors: retry, then try next mirror
                last_err = e
                print(f"⚠️ Download failed ({e}); retrying...")
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch dataset from all mirrors {urls}: {last_err}")


def _validate_csv(path):
    size = os.path.getsize(path)
    if size < MIN_FILE_BYTES:
        raise ValueError(f"Downloaded CSV suspiciously small ({size} bytes); likely truncated.")
    df = pd.read_csv(path, nrows=5)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV schema validation failed; missing columns {sorted(missing)}; got {list(df.columns)}")
    # Full read for row-count check
    n = sum(1 for _ in open(path, 'rb')) - 1
    if n < MIN_ROWS:
        raise ValueError(f"CSV has only {n} rows (< {MIN_ROWS}); likely incomplete.")
    return True


def download_and_convert():
    print("🌍 Connecting to CERN Open Data Portal (Run2011A DoubleMu, 7 TeV)...")

    if not os.path.exists(CSV_FILE):
        print("📥 Downloading educational Dimuon sample (~14MB, 100k events)...")
        _download_with_retry(CSV_FILE)
        print(f"🎉 Acquired CSV payload -> {CSV_FILE}")
    else:
        print(f"✅ CSV '{CSV_FILE}' already present; validating...")

    # Integrity + schema validation: a truncated/corrupt file must FAIL LOUDLY,
    # never be treated as valid for downstream training.
    _validate_csv(CSV_FILE)
    print("✅ CSV integrity/schema check passed.")

    if not os.path.exists(ROOT_FILE):
        print("⚙️ Converting to CERN-native '.root' (TTree 'Events')...")
        df = pd.read_csv(CSV_FILE)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns {sorted(missing)}")
        df = df.dropna(subset=list(REQUIRED_COLUMNS))
        if len(df) < MIN_ROWS:
            raise ValueError(f"Too few valid rows after dropna: {len(df)}")

        # Convert pandas dataframe into an uproot compatible dictionary representation
        # Exclude non-numeric 'Type' constraints (e.g. string 'G') that crash Awkward arrays
        numeric_df = df.select_dtypes(include=[np.number])
        branch_dict = {col: numeric_df[col].to_numpy() for col in numeric_df.columns}

        # Write to ROOT format
        with uproot.recreate(ROOT_FILE) as file:
            file["Events"] = branch_dict

        print(f"✅ ROOT file materialised at: {ROOT_FILE}")
    else:
        print(f"✅ ROOT binary '{ROOT_FILE}' already exists. Ready for ingestion.")


if __name__ == "__main__":
    download_and_convert()
