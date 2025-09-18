#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script generates plots for the scalability analysis (Experiment 2).
It takes a CSV file produced by the `scalability` binary as input and
creates two PDF plots:
1. Peak Memory Usage (MB) vs. Problem Dimension (n)
2. Wall-Clock Time (s) vs. Problem Dimension (n)

The plots compare the performance of the 'standard' and 'two-pass' Lanczos
variants as the problem size increases.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import os

# # Apply the 'science' and 'ieee' styles for high-quality plots.
# # Disable TeX rendering to avoid dependency on a LaTeX installation.
# plt.style.use(['science', 'ieee'])
# plt.rcParams.update({
#     "text.usetex": False,
# })

def plot_memory_vs_dimension(df, output_path):
    """
    Generates a plot of Peak Memory Usage vs. Problem Dimension (n).

    Args:
        df (pd.DataFrame): The input data.
        output_path (str): The path to save the output PDF file.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Convert KB to MB for better readability
    df['rss_mb'] = df['rss_kb'] / 1024

    for variant in df['variant'].unique():
        subset = df[df['variant'] == variant]
        ax.plot(subset['n'], subset['rss_mb'], marker='o', linestyle='-', label=variant.replace('-', ' ').title())

    ax.set_xlabel('Problem Dimension (n)')
    ax.set_ylabel('Peak Memory Usage (MB)')
    ax.set_title('Memory Scalability')
    ax.legend(title='Lanczos Variant')
    ax.grid(True)

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated plot: {output_path}")

def plot_time_vs_dimension(df, output_path):
    """
    Generates a plot of Wall-Clock Time vs. Problem Dimension (n).

    Args:
        df (pd.DataFrame): The input data.
        output_path (str): The path to save the output PDF file.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for variant in df['variant'].unique():
        subset = df[df['variant'] == variant]
        ax.plot(subset['n'], subset['time_s'], marker='o', linestyle='-', label=variant.replace('-', ' ').title())

    ax.set_xlabel('Problem Dimension (n)')
    ax.set_ylabel('Wall-Clock Time (s)')
    ax.set_title('Time Scalability')
    ax.legend(title='Lanczos Variant')
    ax.grid(True)

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated plot: {output_path}")

def main():
    """
    Main function to parse arguments, read data, and generate plots.
    """
    parser = argparse.ArgumentParser(description='Plot scalability analysis results.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file from the scalability runner.')
    parser.add_argument('--output-prefix', type=str, required=True, help='Prefix for the output PDF plot files.')

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate the plots
    plot_memory_vs_dimension(df, f"{args.output_prefix}_memory.pdf")
    plot_time_vs_dimension(df, f"{args.output_prefix}_time.pdf")

if __name__ == '__main__':
    main()
