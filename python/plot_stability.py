#!/usr/bin/env python3

"""
Generates a accuracy plot for a single Lanczos experiment.

This script takes a CSV file produced by the `accuracy-runner` binary and
creates a two-panel figure:
1. The relative error of both Lanczos methods vs. a ground-truth solution.
2. The direct L2-norm of the deviation between the two method's solutions.

The script automatically infers plot titles from the input filename.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_title(input_filename: str) -> str:
    """
    Generates a plot title based on keywords in the filename.
    """
    input_filename = input_filename.lower()
    # Use raw strings with LaTeX for mathematical notation
    if "exp" in input_filename:
        func_part = r"Matrix Exponential $f(z) = \exp(z)$"
    elif "inv" in input_filename:
        func_part = r"Matrix Inverse $f(z) = z^{-1}$"
    else:
        func_part = "Matrix Function"

    if "well-conditioned" in input_filename:
        cond_part = "(Well-Conditioned)"
    elif "ill-conditioned" in input_filename:
        cond_part = "(Ill-Conditioned)"
    else:
        cond_part = ""

    return f"Accuracy for {func_part} {cond_part}".strip()


def create_plot(input_path: str, output_path: str):
    """
    Creates a two-panel figure for a single experiment.

    # Arguments
    * `input_path` - The path to the input CSV data file.
    * `output_path` - The path to save the output PDF file.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        return
    except pd.errors.EmptyDataError:
        print(f"Warning: Input file is empty, skipping plot: '{input_path}'")
        return

    # Automatically generate the title from the input filename.
    title = generate_title(Path(input_path).name)

    # Enable LaTeX rendering for matplotlib
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 12), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # --- Subplot 1: Relative Error vs. Ground Truth ---
    ax1.plot(
        df["k"], df["relative_error_standard"],
        label="Standard Lanczos (One-Pass)",
        marker='o', linestyle='-', color='C0'
    )
    ax1.plot(
        df["k"], df["relative_error_two_pass"],
        label="Two-Pass Lanczos",
        marker='x', linestyle='--', color='C1'
    )
    ax1.set_ylabel("Relative Error vs. Ground Truth", fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_title("Convergence to True Solution", fontsize=14)

    # --- Subplot 2: Direct Solution Deviation ---
    ax2.plot(
        df["k"], df["relative_solution_deviation"],
        # Use a raw string for the LaTeX label
        label=r"Relative Deviation $\|x_k - x'_k\|_2 / \|x_k\|_2$",
        marker='.', linestyle=':', color='C2'
    )
    ax2.set_xlabel("Number of Iterations (k)", fontsize=12)
    ax2.set_ylabel("Relative Solution Deviation", fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_title("Numerical Equivalence of Methods", fontsize=14)

    # Ensure the output directory exists.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Successfully saved plot to: {output_path}")
    plt.close(fig)

def main():
    """Parses command-line arguments and runs the plotting function."""
    parser = argparse.ArgumentParser(
        description="Plot results for a single Lanczos numerical accuracy experiment."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV data file from the accuracy runner."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for the output PDF plot file."
    )
    args = parser.parse_args()
    create_plot(args.input, args.output)

if __name__ == "__main__":
    main()
