#!/usr/bin/env python3

"""
Generates plots for the Lanczos memory-computation trade-off experiment.

This script reads a CSV file with aggregated statistical data produced by the
`tradeoff.rs` runner and creates two publication-quality plots:
1. Peak Memory Usage (RSS) vs. Number of Iterations (k).
2. Wall-Clock Time vs. Number of Iterations (k).

The plots include uncertainty bands (median ± 1 stddev) to visualize the
variability of the measurements.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_plots(input_csv_path: str, output_prefix: str):
    """
    Reads aggregated experiment data and generates memory and time plots.

    # Arguments

    * `input_csv_path` - Path to the input CSV file containing the aggregated results.

    * `output_prefix` - The base path and filename for the output plots.
                        For example, 'report/figures/tradeoff' will produce
                        'tradeoff_memory.pdf' and 'tradeoff_time.pdf'.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv_path}'")
        return

    # Ensure the output directory exists.
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Separate dataframes for each variant for easier plotting.
    df_standard = df[df['variant'] == 'standard'].copy()
    df_two_pass = df[df['variant'] == 'two-pass'].copy()

    # Set a professional plot style.
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- 1. Memory Usage Plot ---
    fig_mem, ax_mem = plt.subplots(figsize=(8, 6))

    # Prepare data for the standard (one-pass) variant.
    # Explicitly cast to float and convert to NumPy for plotting.
    k_standard = df_standard["k"].to_numpy()
    rss_median_standard = df_standard["rss_kb_median"].astype(float).to_numpy() / 1024
    rss_stddev_standard = df_standard["rss_kb_stddev"].astype(float).to_numpy() / 1024

    # Prepare data for the two-pass variant.
    k_two_pass = df_two_pass["k"].to_numpy()
    rss_median_two_pass = df_two_pass["rss_kb_median"].astype(float).to_numpy() / 1024
    rss_stddev_two_pass = df_two_pass["rss_kb_stddev"].astype(float).to_numpy() / 1024

    # Plot median line and uncertainty band for the standard variant.
    ax_mem.plot(
        k_standard, rss_median_standard,
        label="Standard Lanczos (One-Pass)",
        marker='o', linestyle='-', color='C0'
    )
    ax_mem.fill_between(
        k_standard,
        rss_median_standard - rss_stddev_standard, # type: ignore
        rss_median_standard + rss_stddev_standard, # type: ignore
        color='C0', alpha=0.2
    )

    # Plot median line and uncertainty band for the two-pass variant.
    ax_mem.plot(
        k_two_pass, rss_median_two_pass,
        label="Two-Pass Lanczos",
        marker='x', linestyle='--', color='C1'
    )
    ax_mem.fill_between(
        k_two_pass,
        rss_median_two_pass - rss_stddev_two_pass, # type: ignore
        rss_median_two_pass + rss_stddev_two_pass, # type: ignore
        color='C1', alpha=0.2
    )

    ax_mem.set_xlabel("Number of Iterations (k)", fontsize=12)
    ax_mem.set_ylabel("Peak Memory Usage (MB)", fontsize=12)
    ax_mem.set_title("Memory Usage vs. Iteration Count", fontsize=14, fontweight='bold')
    ax_mem.legend(fontsize=11)
    ax_mem.tick_params(axis='both', which='major', labelsize=10)
    ax_mem.grid(True, which='both', linestyle='--', linewidth=0.5)

    memory_plot_path = f"{output_prefix}_memory.pdf"
    fig_mem.savefig(memory_plot_path, bbox_inches='tight')
    print(f"Successfully saved memory plot to: {memory_plot_path}")
    plt.close(fig_mem)

    # --- 2. Execution Time Plot ---
    fig_time, ax_time = plt.subplots(figsize=(8, 6))

    # Prepare data for plotting.
    time_median_standard = df_standard["time_s_median"].astype(float).to_numpy()
    time_stddev_standard = df_standard["time_s_stddev"].astype(float).to_numpy()

    time_median_two_pass = df_two_pass["time_s_median"].astype(float).to_numpy()
    time_stddev_two_pass = df_two_pass["time_s_stddev"].astype(float).to_numpy()

    # Plot median line and uncertainty band for the standard variant.
    ax_time.plot(
        k_standard, time_median_standard,
        label="Standard Lanczos (One-Pass)",
        marker='o', linestyle='-', color='C0'
    )
    ax_time.fill_between(
        k_standard,
        time_median_standard - time_stddev_standard, # type: ignore
        time_median_standard + time_stddev_standard, # type: ignore
        color='C0', alpha=0.2
    )

    # Plot median line and uncertainty band for the two-pass variant.
    ax_time.plot(
        k_two_pass, time_median_two_pass,
        label="Two-Pass Lanczos",
        marker='x', linestyle='--', color='C1'
    )
    ax_time.fill_between(
        k_two_pass,
        time_median_two_pass - time_stddev_two_pass, # type: ignore
        time_median_two_pass + time_stddev_two_pass, # type: ignore
        color='C1', alpha=0.2
    )

    ax_time.set_xlabel("Number of Iterations (k)", fontsize=12)
    ax_time.set_ylabel("Wall-Clock Time (seconds)", fontsize=12)
    ax_time.set_title("Execution Time vs. Iteration Count", fontsize=14, fontweight='bold')
    ax_time.legend(fontsize=11)
    ax_time.tick_params(axis='both', which='major', labelsize=10)
    ax_time.grid(True, which='both', linestyle='--', linewidth=0.5)

    time_plot_path = f"{output_prefix}_time.pdf"
    fig_time.savefig(time_plot_path, bbox_inches='tight')
    print(f"Successfully saved time plot to: {time_plot_path}")
    plt.close(fig_time)

def main():
    """Parses command-line arguments and runs the plotting function."""
    parser = argparse.ArgumentParser(
        description="Plot results for the Lanczos trade-off experiment."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file generated by the tradeoff runner."
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Prefix for the output PDF plot files (e.g., 'report/figures/tradeoff')."
    )
    args = parser.parse_args()
    create_plots(args.input, args.output_prefix)

if __name__ == "__main__":
    main()
