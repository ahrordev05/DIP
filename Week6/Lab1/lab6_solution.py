from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import color, exposure, io


ROOT_DIR = Path(__file__).resolve().parents[2]
WEEK6_DIR = Path(__file__).resolve().parents[1]
LAB1_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LAB1_DIR / "outputs"
INPUT_IMAGE = ROOT_DIR / "Week5" / "lab1" / "img.png"


def box_filt(n: int) -> np.ndarray:
    if n <= 0 or n % 2 == 0:
        raise ValueError("Window size must be a positive odd integer.")
    return np.ones((n, n), np.float32) / (n * n)


def gauss_filt(n: int, sigma: float = 1.0) -> np.ndarray:
    if n <= 0 or n % 2 == 0:
        raise ValueError("Window size must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    ax = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def apply_filters(image_input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    n = kernel.shape[0]
    pad = n // 2
    padded = np.pad(image_input, ((pad, pad), (pad, pad)), mode="symmetric")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (n, n))
    filtered = np.einsum("ijkl,kl->ij", windows, kernel, dtype=np.float32)
    return filtered.astype(np.float32)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_input_gray() -> np.ndarray:
    image = io.imread(str(INPUT_IMAGE))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {INPUT_IMAGE}")

    if image.ndim == 2:
        gray = image.astype(np.float32)
        if gray.max() > 1.0:
            gray /= 255.0
        return gray

    if image.shape[2] == 4:
        image = color.rgba2rgb(image)
    gray = color.rgb2gray(image).astype(np.float32)
    return gray


def save_task1(gray: np.ndarray) -> tuple[np.ndarray, str]:
    equalized = exposure.equalize_hist(gray)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    ax[0, 0].set_title("Original (grayscale)")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(equalized, cmap="gray", vmin=0, vmax=1)
    ax[0, 1].set_title("Histogram Equalized")
    ax[0, 1].axis("off")

    ax[1, 0].hist(gray.ravel(), bins=256, range=(0, 1), color="steelblue")
    ax[1, 0].set_title("Histogram: Original")
    ax[1, 0].set_xlabel("Intensity")
    ax[1, 0].set_ylabel("Count")

    ax[1, 1].hist(equalized.ravel(), bins=256, range=(0, 1), color="tomato")
    ax[1, 1].set_title("Histogram: Equalized")
    ax[1, 1].set_xlabel("Intensity")
    ax[1, 1].set_ylabel("Count")

    fig.suptitle("Task 1 - Histogram Equalization")
    fig.tight_layout()

    out_path = OUTPUT_DIR / "task1_hist_equalization.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return equalized, str(out_path)


def save_task2(gray: np.ndarray) -> dict[str, np.ndarray]:
    results = {
        "Original": gray,
        "Box 5x5": np.clip(apply_filters(gray, box_filt(5)), 0, 1),
        "Box 9x9": np.clip(apply_filters(gray, box_filt(9)), 0, 1),
        "Gaussian 3x3 (sigma=0.8)": np.clip(apply_filters(gray, gauss_filt(3, 0.8)), 0, 1),
        "Gaussian 5x5 (sigma=1.0)": np.clip(apply_filters(gray, gauss_filt(5, 1.0)), 0, 1),
        "Gaussian 9x9 (sigma=2.0)": np.clip(apply_filters(gray, gauss_filt(9, 2.0)), 0, 1),
        "Gaussian 5x5 (sigma=1.5)": np.clip(apply_filters(gray, gauss_filt(5, 1.5)), 0, 1),
        "Gaussian 5x5 (sigma=3.0)": np.clip(apply_filters(gray, gauss_filt(5, 3.0)), 0, 1),
    }

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for i, (title, img) in enumerate(results.items()):
        r, c = divmod(i, 4)
        ax[r, c].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[r, c].set_title(title, fontsize=10)
        ax[r, c].axis("off")
    fig.suptitle("Task 2 - Box vs Gaussian Filtering")
    fig.tight_layout()

    out_path = OUTPUT_DIR / "task2_box_gaussian_comparison.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return results


def save_task3(gray: np.ndarray) -> dict[str, np.ndarray]:
    blurred = np.clip(apply_filters(gray, gauss_filt(9, 10.0)), 0, 1)
    amounts = [0.5, 1.0, 1.5, 2.0]

    outputs: dict[str, np.ndarray] = {"Original": gray, "Blurred (9x9, sigma=10)": blurred}
    for amt in amounts:
        sharpened = np.clip(gray * (1.0 + amt) - blurred * amt, 0, 1)
        outputs[f"Unsharp amount={amt}"] = sharpened

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for i, (title, img) in enumerate(outputs.items()):
        r, c = divmod(i, 3)
        ax[r, c].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[r, c].set_title(title, fontsize=10)
        ax[r, c].axis("off")
    fig.suptitle("Task 3 - Unsharp Masking with Different Weights")
    fig.tight_layout()

    out_path = OUTPUT_DIR / "task3_unsharp_levels.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return outputs


def save_task4(gray: np.ndarray) -> list[dict[str, float]]:
    filt_sizes = [3, 5, 7, 9, 11, 15]
    k_values = [-2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 9]

    gray_f = gray.astype(np.float32)
    rows: list[dict[str, float]] = []

    lap_matrix = np.zeros((len(filt_sizes), len(k_values)), dtype=np.float32)
    clip_matrix = np.zeros((len(filt_sizes), len(k_values)), dtype=np.float32)

    for i, n in enumerate(filt_sizes):
        blur = apply_filters(gray_f, box_filt(n))
        diff = gray_f - blur
        for j, k in enumerate(k_values):
            sharp_raw = gray_f + k * diff
            clipped = np.clip(sharp_raw, 0, 1)
            lap_var = float(np.var(ndi.laplace(clipped)))
            mse = float(np.mean((clipped - gray_f) ** 2))
            clip_fraction = float(np.mean((sharp_raw < 0) | (sharp_raw > 1)))

            lap_matrix[i, j] = lap_var
            clip_matrix[i, j] = clip_fraction

            rows.append(
                {
                    "filt_size": int(n),
                    "k": float(k),
                    "laplacian_var": lap_var,
                    "mse_vs_original": mse,
                    "clip_fraction": clip_fraction,
                }
            )

    csv_path = OUTPUT_DIR / "task4_parameter_study.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filt_size", "k", "laplacian_var", "mse_vs_original", "clip_fraction"]
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    im0 = ax[0].imshow(lap_matrix, cmap="viridis", aspect="auto")
    ax[0].set_title("Laplacian Variance (higher = sharper)")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("filt_size")
    ax[0].set_xticks(np.arange(len(k_values)))
    ax[0].set_xticklabels([str(k) for k in k_values], rotation=45, ha="right")
    ax[0].set_yticks(np.arange(len(filt_sizes)))
    ax[0].set_yticklabels([str(n) for n in filt_sizes])
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(clip_matrix, cmap="magma", aspect="auto")
    ax[1].set_title("Clipping Fraction (lower is safer)")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("filt_size")
    ax[1].set_xticks(np.arange(len(k_values)))
    ax[1].set_xticklabels([str(k) for k in k_values], rotation=45, ha="right")
    ax[1].set_yticks(np.arange(len(filt_sizes)))
    ax[1].set_yticklabels([str(n) for n in filt_sizes])
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    fig.suptitle("Task 4 - Parameter Sweep Using Custom Box-Blur Unsharp Masking")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "task4_heatmaps.png", dpi=180)
    plt.close(fig)

    sample_sizes = [3, 9, 15]
    sample_k = [-1, 0, 1, 3, 5]
    fig, ax = plt.subplots(len(sample_sizes), len(sample_k), figsize=(14, 8))
    for i, n in enumerate(sample_sizes):
        blur = apply_filters(gray_f, box_filt(n))
        diff = gray_f - blur
        for j, k in enumerate(sample_k):
            sharp = np.clip(gray_f + k * diff, 0, 1)
            ax[i, j].imshow(sharp, cmap="gray", vmin=0, vmax=1)
            ax[i, j].set_title(f"n={n}, k={k}", fontsize=9)
            ax[i, j].axis("off")
    fig.suptitle("Task 4 - Sample Visual Results")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "task4_sample_grid.png", dpi=180)
    plt.close(fig)

    return rows


def write_report(task4_rows: list[dict[str, float]]) -> None:
    safe_rows = [r for r in task4_rows if r["clip_fraction"] <= 0.05 and r["k"] >= 0]
    if safe_rows:
        best_safe = max(safe_rows, key=lambda r: r["laplacian_var"])
        best_safe_text = (
            f"Best sharpness with <=5% clipping: filt_size={int(best_safe['filt_size'])}, "
            f"k={best_safe['k']}, laplacian_var={best_safe['laplacian_var']:.2f}, "
            f"clip_fraction={best_safe['clip_fraction']:.4f}"
        )
    else:
        best_safe_text = "No non-negative k setting stayed below 5% clipping."

    aggressive_rows = [r for r in task4_rows if r["k"] >= 3]
    avg_clip_aggressive = np.mean([r["clip_fraction"] for r in aggressive_rows]) if aggressive_rows else 0.0

    report = f"""# Week 6 - Lab Solution (using Week5 image)

- Input image: `{INPUT_IMAGE}`
- Output folder: `{OUTPUT_DIR}`

## Task 1 - Histogram equalization
- The equalized image spreads intensity values across a wider range.
- Histogram after equalization is flatter and occupies more bins, which improves global contrast.
- This helps in poor lighting because dark/bright regions that were packed into narrow intensity ranges become more distinguishable.
- Figure: `task1_hist_equalization.png`

## Task 2 - Gaussian filter
- Box filtering blurs uniformly and tends to smear edges more aggressively.
- Gaussian filtering preserves structures better because center pixels get higher weight than distant neighbors.
- Larger kernels and larger sigma values increase smoothing (noise reduction) but remove more fine detail.
- Figure: `task2_box_gaussian_comparison.png`

## Task 3 - Unsharp masking
- Increasing unsharp amount increases edge contrast and perceived sharpness.
- Small amounts (around 0.5 to 1.5) sharpen details with fewer artifacts.
- Very large amounts can amplify noise and halos around strong edges.
- Figure: `task3_unsharp_levels.png`

## Task 4 - Parameter study (custom unsharp masking)
- Sweep used: `filt_size = [3,5,7,9,11,15]`, `k = [-2,-1,-0.5,0,0.5,1,2,3,5,9]`
- Larger `filt_size` increases the blur component, so for the same `k` the sharpening effect is stronger but less local.
- Negative `k` values blur the image further; `k=0` returns the original.
- High positive `k` gives stronger sharpening but increases clipping and artifact risk.
- {best_safe_text}
- Average clipping for aggressive settings (`k >= 3`): {avg_clip_aggressive:.4f}
- Figures: `task4_heatmaps.png`, `task4_sample_grid.png`
- CSV metrics: `task4_parameter_study.csv`
"""

    (LAB1_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_output_dir()
    gray = load_input_gray()

    save_task1(gray)
    save_task2(gray)
    save_task3(gray)
    task4_rows = save_task4(gray)
    write_report(task4_rows)

    print(f"Done. Outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
