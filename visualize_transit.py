import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

print("Downloading Kepler-8 data (Quarters 4-7)...")
search_result = lk.search_lightcurve('Kepler-8', mission='Kepler', quarter=[4, 5, 6, 7])
lc_collection = search_result.download_all()

# 1. ASYMMETRIC OUTLIER REMOVAL: Strip positive cosmic rays, preserve negative transit dips
print("Flattening, removing outliers, and stitching...")
cleaned_lcs = [
    lc.remove_nans().flatten(window_length=101).remove_outliers(sigma_upper=3, sigma_lower=np.inf) 
    for lc in lc_collection
]
clean_lc = lk.LightCurveCollection(cleaned_lcs).stitch()

# 2. AUTO-ALIGNMENT: Run BLS algorithm to computationally find exact epoch
print("Running BLS algorithm to find exact transit alignment...")
# We sweep a tight search window around Kepler-8b's known ~3.522 day period
periodogram = clean_lc.to_periodogram(method='bls', period=np.arange(3.5, 3.55, 0.0001))
best_period = periodogram.period_at_max_power
best_epoch = periodogram.transit_time_at_max_power

print(f"Algorithm found Period: {best_period.value:.4f} days, Epoch: {best_epoch.value:.4f}")

# 3. Fold the data using the exact BLS parameters
folded_lc = clean_lc.fold(period=best_period, epoch_time=best_epoch)

# 4. Bin the data to average out the remaining Gaussian noise
binned_lc = folded_lc.bin(time_bin_size=0.01)

# 5. Plot the final extraction
fig, ax = plt.subplots(figsize=(10, 5))
folded_lc.scatter(ax=ax, color='gray', s=2, alpha=0.2, label='Raw Folded Flux (Noise)')
binned_lc.scatter(ax=ax, color='blue', s=20, label='Binned Flux (Extracted Signal)')

ax.set_title("In-Distribution Baseline: Binned & Auto-Aligned Transit of Kepler-8b")
ax.set_xlabel("Phase (Days)")
ax.set_ylabel("Normalized Brightness")

# Zoom in tightly around the transit window
plt.xlim(-0.5, 0.5) 
plt.grid(True)
plt.legend(loc='lower right') # Moved legend to avoid tkinter warning
plt.show()