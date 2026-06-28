"""Run the full pipeline end to end and reproduce every result and figure.

Usage:
    cd src && python run_all.py

Steps: download data -> preprocess -> statistics & clustering -> models -> figures -> interpretation.
Outputs land in ../results and ../figures.
"""
import download_data, prep, stats_cluster, models, figures, interpret


def main():
    print("\n=== 1/6 Download data ==="); download_data.main()
    print("\n=== 2/6 Preprocess ==="); prep.main()
    print("\n=== 3/6 Statistics & clustering ==="); stats_cluster.main()
    print("\n=== 4/6 Models & evaluation ==="); models.main()
    print("\n=== 5/6 Figures ==="); figures.main()
    print("\n=== 6/6 Interpretation ==="); interpret.main()
    print("\nDone. See ../results and ../figures.")


if __name__ == "__main__":
    main()
