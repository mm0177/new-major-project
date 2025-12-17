import pandas as pd
import glob
import os

def merge_trade_files():
    # 1. Find all CSV files that look like TradeData
    all_files = glob.glob("TradeData*.csv")
    
    if not all_files:
        print("❌ No 'TradeData*.csv' files found in the current folder.")
        return

    print(f"Found {len(all_files)} files: {all_files}")
    
    combined_frames = []
    
    for f in all_files:
        try:
            # FIX: index_col=False prevents the "Column Shift" error
            # dtype=str ensures we read codes as text (preserving '01' instead of 1)
            df = pd.read_csv(f, index_col=False, dtype=str)
            
            # 2. Filter for Monthly Data Only
            # (Your 2025 file has both Annual 'A' and Monthly 'M' data mixed)
            if 'freqCode' in df.columns:
                df = df[df['freqCode'] == 'M']
            
            # Select only what we need
            # We need: Date (period), Country (reporterDesc), Value (primaryValue)
            if 'period' in df.columns and 'primaryValue' in df.columns:
                subset = df[['period', 'reporterDesc', 'primaryValue']].copy()
                combined_frames.append(subset)
            else:
                print(f"⚠️ Skipped {f}: Missing 'period' or 'primaryValue' column.")
                
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")

    if not combined_frames:
        print("No valid data found to merge.")
        return

    # 3. Combine All
    df_clean = pd.concat(combined_frames, ignore_index=True)
    
    # Rename columns for clarity
    df_clean.columns = ['date_code', 'country', 'trade_value_usd']
    
    # 4. Convert Date (Robustly)
    # This turns '202301' into '2023-01-01'
    print("Converting dates...")
    df_clean['date'] = pd.to_datetime(df_clean['date_code'], format='%Y%m', errors='coerce')
    
    # Drop rows where date failed (e.g. if any Annual data sneaked in)
    df_clean = df_clean.dropna(subset=['date'])
    
    # 5. Clean Trade Value (Convert to float)
    df_clean['trade_value_usd'] = pd.to_numeric(df_clean['trade_value_usd'], errors='coerce')
    
    # 6. Sort and Save
    df_clean = df_clean.sort_values(by=['date', 'country'])
    
    print("✅ Merging Complete!")
    print(f"Total Rows: {len(df_clean)}")
    print(f"Date Range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
    
    output_file = "master_trade_data.csv"
    df_clean.to_csv(output_file, index=False)
    print(f"Saved to '{output_file}'")

if __name__ == "__main__":
    merge_trade_files()