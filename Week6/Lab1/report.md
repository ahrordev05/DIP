# Week 6 - Lab Solution (using Week5 image)

- Input image: `/Users/ahrornosirov/Downloads/DIP/Week5/lab1/img.png`
- Output folder: `/Users/ahrornosirov/Downloads/DIP/Week6/Lab1/outputs`

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
- Best sharpness with <=5% clipping: filt_size=3, k=3.0, laplacian_var=0.06, clip_fraction=0.0276
- Average clipping for aggressive settings (`k >= 3`): 0.1717
- Figures: `task4_heatmaps.png`, `task4_sample_grid.png`
- CSV metrics: `task4_parameter_study.csv`
