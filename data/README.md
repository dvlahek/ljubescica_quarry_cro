# Data

Raw Sentinel-2 Level-2A scenes are not stored in this repository. The reconstruction script downloads them through Sentinel Hub after local credentials are set.

Generated outputs are written to `quarry_timelapse_science_qc_v8/` and are ignored by Git by default.

For the supervised benchmark, the two endpoint masks used as manual references are expected to be available as PNG pairs with the following names in the data directory used by the benchmark script:

```text
2017-07_visual_rgb.png
2017-07_mask.png
2025-08_visual_rgb.png
2025-08_mask.png
```

Intermediate workflow-derived masks are treated as pseudo-labels, not independent ground truth.
