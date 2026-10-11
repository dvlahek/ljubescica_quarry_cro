# Ljubeščica quarry Sentinel-2 monitoring

Reproducibility code for the manuscript:

**Quarry expansion and forest-edge spectral change detected from Sentinel-2 time series: An eight-year land-cover degradation case study in northern Croatia**

The repository contains a minimal public workflow for reconstructing the Sentinel-2 processing pipeline and the limited RF/SVM benchmark used in the revised manuscript. No Sentinel Hub credentials are stored in the repository.

## What is included

```text
scripts/reconstruct_article_ljubescica.py
scripts/benchmark_rf_svm_manual_and_pseudo.py
requirements.txt
```

The reconstruction script downloads Sentinel-2 Level-2A data for the Ljubeščica quarry analysis window, computes NDVI, BSI and NDBI, applies the calibrated quarry-mask workflow, exports monthly metrics, and generates basic plots.

The benchmark script runs two checks:

1. a non-circular manual cross-year RF/SVM benchmark using the two manually classified endpoint masks, and
2. a pseudo-label transfer check trained on workflow-derived masks and evaluated on the manual endpoint masks.

The second check is not independent validation. It is included only as a consistency and transfer check.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Sentinel Hub credentials

Create Sentinel Hub OAuth credentials in your Sentinel Hub account and set them as environment variables.

Linux/macOS:

```bash
export SENTINELHUB_CLIENT_ID="your-client-id"
export SENTINELHUB_CLIENT_SECRET="your-client-secret"
```

Windows PowerShell:

```powershell
$env:SENTINELHUB_CLIENT_ID="your-client-id"
$env:SENTINELHUB_CLIENT_SECRET="your-client-secret"
```

Do not commit credentials, `.env` files, tokens, or local configuration files.

## Run the reconstruction workflow

```bash
python scripts/reconstruct_article_ljubescica.py
```

Default output folder:

```text
quarry_timelapse_science_qc_v8/
```

Main outputs:

```text
quarry_science_metrics.csv
run_summary.json
plot_1_area.png
plot_2_fhfm_ndvi.png
plot_3_centroid_shift.png
```

The manuscript statistics are based on the archived result set retained by the authors. A live Sentinel Hub rerun may differ slightly because cloud selection, mosaicking, service versions, and quality-control decisions can change.

## Run the RF/SVM benchmark

After the reconstruction workflow has produced visual RGB images and masks, run:

```bash
python scripts/benchmark_rf_svm_manual_and_pseudo.py --data_dir quarry_timelapse_science_qc_v8
```

Manual endpoint masks should correspond to:

```text
2017-07_mask.png
2025-08_mask.png
```

The benchmark uses archived RGB proxy features from `*_visual_rgb.png` files. It should not be described as a full multi-band supervised validation experiment unless the script is modified to use the complete Sentinel-2 feature stack.

## Data availability

Sentinel-2 Level-2A data are publicly available through Copernicus Data Space Ecosystem and Sentinel Hub. Derived masks, metrics, and generated plots can be produced using the scripts in this repository after setting valid Sentinel Hub credentials.
