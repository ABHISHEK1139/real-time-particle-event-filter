import os
import urllib.request
import pandas as pd
import numpy as np
import uproot

DATA_URL = "http://opendata.cern.ch/record/545/files/Dimuon_DoubleMu.csv"
CSV_FILE = "Dimuon_DoubleMu.csv"
ROOT_FILE = "Dimuon_DoubleMu.root"

def download_and_convert():
    print(f"🌍 Initiating secure connection to CERN Open Data Portal...")
    
    if not os.path.exists(CSV_FILE):
        print(f"📥 Downloading empirical Dimuon scattering data (~14MB)...")
        try:
            urllib.request.urlretrieve(DATA_URL, CSV_FILE)
            print(f"🎉 Successfully acquired CSV payload from: {DATA_URL}")
        except Exception as e:
            print(f"❌ Pipeline Interruption! Failed to fetch data: {str(e)}")
            print("Please manually download the payload from: http://opendata.cern.ch/record/545")
            return
            
    if not os.path.exists(ROOT_FILE):
        print(f"⚙️ Compiling into CERN-Native '.root' Binary (TTree)...")
        df = pd.read_csv(CSV_FILE)
        
        # Convert pandas dataframe into an uproot compatible dictionary representation
        # Exclude non-numeric 'Type' constraints (e.g. string 'G') that crash Awkward arrays
        numeric_df = df.select_dtypes(include=[np.number])
        branch_dict = {col: numeric_df[col].to_numpy() for col in numeric_df.columns}
        
        # Write to ROOT format
        with uproot.recreate(ROOT_FILE) as file:
            file["Events"] = branch_dict
            
        print(f"✅ ROOT file successfully materialized at: {ROOT_FILE}")
    else:
        print(f"✅ Native ROOT binary '{ROOT_FILE}' already exists perfectly. Ready for ingestion.")

if __name__ == "__main__":
    download_and_convert()
