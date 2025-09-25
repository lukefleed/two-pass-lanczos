#!/usr/bin/env python3

"""
Generates a plot for the orthogonality analysis of the Lanczos bases.

This script reads a CSV file produced by the `orthogonality-runner` binary and
creates a plot that visualizes the loss of orthogonality for both the standard
(one-pass) and regenerated (two-pass) Lanczos bases as a function of the
number of iterations.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_title(input_filename: str) -> str:
    """
    Generates a descriptive plot title based on keywords in the input filename.
    """
    input_filename = input_filename.lower()
    func_part = ""
    if "exp" in input_filename:
        func_part = r"for $f(z) = \exp(z)$"
    elif "inv" in input_filename:
        func_part = r"for $f(z) = z^{-1}$"

    cond_part = ""
    if "well-conditioned" in input_filename:
        cond_part = "(Well-Conditioned)"
    elif "ill-conditioned" in input_filename:
        cond_part = "(Ill-Conditioned)"

    return f"Loss of Orthogonality vs. Iteration Count {func_part} {cond_part}".strip()

def create_plot(input_path: str, output_path: str):
    """
    Creates a single-panel figure visualizing the loss of orthogonality.

    # Arguments
    * `input_path` - The path to the input CSV data file.
    * `output_path` - The path where the output PDF file will be saved.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        return
    except pd.errors.EmptyDataError:
        print(f"Warning: Input file '{input_path}' is empty. Skipping plot generation.")
        return

    title = generate_title(Path(input_path).name)

    # Configure matplotlib for publication-quality, LaTeX-rendered plots.
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the orthogonality loss for both the standard and regenerated bases.
    ax.plot(
        df["k"], df["ortho_loss_standard"],
        label=r"Standard Basis $V_k$ (One-Pass)",
        marker='o', linestyle='-', color='C0'
    )
    ax.plot(
        df["k"], df["ortho_loss_regenerated"],
        label=r"Regenerated Basis $V'_k$ (Two-Pass)",
        marker='x', linestyle='--', color='C1'
    )

    ax.set_xlabel("Number of Iterations (k)", fontsize=12)
    ax.set_ylabel(r"Orthogonality Loss $\|I - V_k^H V_k\|_F$", fontsize=12)
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Ensure the output directory exists before attempting to save the file.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path, bbox_inches='tight')
    print(f"Successfully saved orthogonality loss plot to: {output_path}")
    plt.close(fig)

def main():
    """Parses command-line arguments and orchestrates plot generation."""
    parser = argparse.ArgumentParser(
        description="Plot orthogonality loss from a stability experiment CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV data file from the orthogonality runner."
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
