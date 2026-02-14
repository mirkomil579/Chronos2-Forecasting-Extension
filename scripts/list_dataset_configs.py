import argparse
from chronos2_project.data import list_hf_configs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset name (e.g., autogluon/chronos_datasets)")
    args = ap.parse_args()
    configs = list_hf_configs(args.dataset)
    for c in configs:
        print(c)

if __name__ == "__main__":
    main()
