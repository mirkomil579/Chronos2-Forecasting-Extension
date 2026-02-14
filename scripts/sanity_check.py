import argparse
import pandas as pd
from chronos import Chronos2Pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    args = ap.parse_args()

    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)

    # tiny synthetic series
    df = pd.DataFrame({
        "id": [0]*200,
        "timestamp": pd.date_range("2020-01-01", periods=200, freq="H"),
        "target": [float(i) for i in range(200)],
    })
    pred = pipeline.predict_df(df, prediction_length=24, quantiles=[0.1,0.5,0.9])
    print(pred.head())
    print("OK - Chronos-2 loaded and produced predictions.")

if __name__ == "__main__":
    main()
