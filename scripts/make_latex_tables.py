import argparse
import os
import pandas as pd

def df_to_ieee_table(df: pd.DataFrame, caption: str, label: str) -> str:
    # keep it simple; user may adjust formatting
    latex = df.to_latex(index=False, float_format="%.4f")
    latex = latex.replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        latex,
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True, help="Path to metrics.csv")
    ap.add_argument("--out_tex", required=True)
    ap.add_argument("--caption", default="Results")
    ap.add_argument("--label", default="tab:results")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_csv)
    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(df_to_ieee_table(df, args.caption, args.label))

    print(f"Wrote {args.out_tex}")

if __name__ == "__main__":
    main()
