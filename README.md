# TikTok Recommendation Audit

**Tim Laux | DS480**

Analysis code accompanying my case study of political exposure in TikTok recommendations during the 2024 US election. The data come from [Ibrahim et al. (2026)](https://doi.org/10.1038/s41586-026-10447-1).

## Data

Download the March 8, 2026 Code/Data release from [Zenodo](https://doi.org/10.5281/zenodo.18907782). Extract the archive and copy its `data/recommendations.csv` into a `data/` folder beside `tiktok_analysis.py`.

The dataset is downloaded separately.

## Run

Requires Python 3.12. From the repository folder, run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tiktok_analysis.py
```

Results are saved in `analysis_results/`. The main output is `results_full_precision.csv`.

## Methods

The analysis compares 22,397 Democratic- and Republican-conditioned recommendation records. It averages run shares within state-week-condition groups, then observed weeks within each state, then the three states equally.

It includes 20,000 joint-week bootstrap resamples, an adjacent-week sensitivity check, and supportive-only and pooled-record comparisons. Neutral-labelled records remain in denominators, and repeated video deliveries are retained.

Results describe the retained audit records and do not establish effects on voting.

## AI assistance

Codex assisted with code generation and debugging. Because undertaking a full empirical replication was optional for DS480 students, AI assistance was a great help in making a deep dive into the underlying dataset. 
