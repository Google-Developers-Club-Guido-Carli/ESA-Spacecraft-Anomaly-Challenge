"""Quick test to check data loading."""
import pandas as pd

# Try reading train.parquet instead
print("Testing parquet loading...")
try:
    df = pd.read_parquet('../.data/train.parquet')
    print(f"Success! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check for channel columns
    channel_cols = [c for c in df.columns if 'channel' in c.lower()]
    print(f"\nChannel columns found: {len(channel_cols)}")
    print(f"Examples: {channel_cols[:10]}")
    
    # Check for is_anomaly
    if 'is_anomaly' in df.columns:
        print(f"\nis_anomaly found! Rate: {df['is_anomaly'].mean():.6f}")
except Exception as e:
    print(f"Error loading parquet: {e}")
    
    # Try CSV as backup
    print("\nTrying CSV as backup...")
    try:
        df = pd.read_csv('../.data/train.csv', nrows=5)
        print(f"CSV works! Shape: {df.shape}")
    except Exception as e2:
        print(f"CSV also failed: {e2}")
