#!/usr/bin/env python3

"""
Calculates the memory growth rate from a scalability experiment CSV file.

This script reads a CSV file containing scalability results, calculates the
difference in peak memory usage (in bytes) between the 'standard' and
'two-pass' Lanczos variants, and performs a linear regression to determine
the memory growth rate per 1000 units of the problem dimension 'n'.
"""

import argparse
import pandas as pd
from scipy import stats

def calculate_growth_rate(filepath: str):
    """
    Processes the CSV file and prints the calculated memory growth rate.

    # Arguments
    * `filepath` - Path to the input CSV data file.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{filepath}'")
        return
    except pd.errors.EmptyDataError:
        print(f"Warning: Input file '{filepath}' is empty. Cannot perform calculation.")
        return

    # Separate standard and two-pass data
    df_standard = df[df['variant'] == 'standard'][['n', 'rss_kb']].rename(columns={'rss_kb': 'standard_rss_kb'})
    df_two_pass = df[df['variant'] == 'two-pass'][['n', 'rss_kb']].rename(columns={'rss_kb': 'two_pass_rss_kb'})

    # Merge the data on 'n' to align corresponding measurements
    merged_df = pd.merge(df_standard, df_two_pass, on='n')

    # Calculate the difference in memory usage in Kilobytes for each n
    merged_df['rss_kb_diff'] = merged_df['standard_rss_kb'] - merged_df['two_pass_rss_kb']

    # Convert the difference from Kilobytes to Bytes for the regression
    merged_df['rss_bytes_diff'] = merged_df['rss_kb_diff'] * 1024

    # The independent variable (x) is the problem dimension 'n'
    x_data = merged_df['n'].values
    # The dependent variable (y) is the memory difference in bytes
    y_data = merged_df['rss_bytes_diff'].values

    # Perform linear regression: y = slope * x + intercept
    lin_reg_result = stats.linregress(x=x_data, y=y_data)
    slope = lin_reg_result[0]  # slope is the first element of the result tuple

    # Calculate the growth rate per 1000 units of n.
    growth_rate_per_1000_n = slope * 1000  # type: ignore

    print("--- Linear Regression Results ---")
    print(f"Slope (bytes per unit increase in n): {slope:.4f}")
    print(f"\nGrowth rate per 1000 units of n: {growth_rate_per_1000_n:.2f} bytes")
    print("\n--- Raw Data Table (in MB) ---")

    # Create a display-friendly table in Megabytes
    df_display = pd.DataFrame()
    df_display['Dimension (n)'] = merged_df['n']
    df_display['Standard (MB)'] = (merged_df['standard_rss_kb'] / 1024).round(1)
    df_display['Two-Pass (MB)'] = (merged_df['two_pass_rss_kb'] / 1024).round(1)
    df_display['Difference (MB)'] = (merged_df['rss_kb_diff'] / 1024).round(1)
    print(df_display.to_string(index=False))


def main():
    """Parses command-line arguments and runs the calculation."""
    parser = argparse.ArgumentParser(
        description="Calculate memory growth rate from scalability data."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV data file."
    )
    args = parser.parse_args()
    calculate_growth_rate(args.input)


if __name__ == "__main__":
    main()
