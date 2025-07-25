<div align="justify">

# <div align="center">Sonar Dataset Documentation</div>

## Overview

The `sonar_data.csv` file contains the Sonar Dataset, a classic benchmark for binary classification in machine learning. The dataset consists of sonar signal returns collected from a metal cylinder (representing a mine) and a roughly cylindrical rock on the sea floor. The goal is to classify each observation as either a rock (R) or a mine (M) based on the energy of sonar signals at various frequencies.

- **Total Samples:** 208
- **Features:** 60 continuous, real-valued attributes
- **Target Classes:** 2 (Rock = R, Mine = M)
- **Source:** [UCI Machine Learning Repository - Connectionist Bench (Sonar, Mines vs. Rocks) Data Set](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+%28sonar,+mines+vs+rocks%29)

## Data Structure

Each row in the CSV file represents a single sonar observation. The columns are as follows:

- **Columns 0-59:** Energy values of sonar signals in 60 frequency bands (attributes)
- **Column 60:** Target label ('R' for Rock, 'M' for Mine)

All feature values are normalized to the range [0, 1].

### Example Row

```
0.0200,0.0371,0.0428,0.0207,0.0954,...,0.0032,R
```

## Feature Description

- **Attributes 0-59:**

  - Each attribute represents the energy within a specific frequency band, integrated over a certain period of time.
  - The frequency bands are ordered from low to high.
  - All values are continuous and normalized.
  - No missing values are present.

- **Target (Column 60):**
  - 'R' = Rock
  - 'M' = Mine

## Statistical Properties

- **Balanced Classes:** The dataset contains approximately equal numbers of rocks and mines.
- **No Missing Data:** All entries are complete.
- **Normalization:** All features are scaled between 0 and 1.

### Class Distribution

- **R (Rock):** ~111 samples
- **M (Mine):** ~97 samples

### Feature Statistics (Typical Ranges)

- **Minimum:** 0.0
- **Maximum:** 1.0
- **Mean:** Varies by feature, typically between 0.03 and 0.25
- **Standard Deviation:** Varies by feature, typically between 0.05 and 0.2

## Usage Notes

- **Machine Learning:** Commonly used for binary classification tasks, especially for testing algorithms' ability to distinguish subtle patterns in real-world data.
- **Preprocessing:** No additional preprocessing is required; the data is ready for use.
- **Interpretation:** Each feature is a spectral energy measurement; higher values indicate stronger sonar returns in that frequency band.

## References

- UCI Machine Learning Repository: [Sonar, Mines vs. Rocks Data Set](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+%28sonar,+mines+vs+rocks%29)
- Gorman, R. P., & Sejnowski, T. J. (1988). Analysis of hidden units in a layered network trained to classify sonar targets. Neural Networks, 1(1), 75-89.

## File Format

- **File Name:** `sonar_data.csv`
- **Format:** Comma-separated values (CSV)
- **Rows:** 208
- **Columns:** 61 (60 features + 1 label)

</div>

---

<div align="center">

For further details on how the data is used in this project, see the main notebook [rock_mine_prediction.ipynb](https://github.com/NhanPhamThanh-IT/Rock-Mine-Logistic/blob/main/rock_mine_prediction.ipynb) and the [README](https://github.com/NhanPhamThanh-IT/Rock-Mine-Logistic/blob/main/README.md).

</div>
