import pandas as pd

CHANNELS = [f"channel_{i}" for i in range(41, 47)]

def to_timeeval_csv(in_path, out_path, has_labels):
    df = pd.read_csv(in_path)
    df = df.rename(columns={"id": "timestamp"})  
    cols = ["timestamp"] + CHANNELS + (["is_anomaly"] if has_labels and "is_anomaly" in df.columns else [])
    df = df[cols]
    df.to_csv(out_path, index=False)


to_timeeval_csv("/content/drive/MyDrive/telemanom_esa/data/train_lite.csv", "/content/drive/MyDrive/telemanom_esa/data/train_te.csv", has_labels=True)
to_timeeval_csv("/content/drive/MyDrive/telemanom_esa/data/test_lite.csv",  "/content/drive/MyDrive/telemanom_esa/data/test_te.csv",  has_labels=False)  

