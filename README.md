# TikTok Political Recommendation Audit (2024 Election)

**Author:** Tim Laux
**Course:** DS480

This repository contains the replication script and analytical pipeline for auditing algorithmic recommendation skews on TikTok during the 2024 US Presidential Election, using the audit dataset from Ibrahim et al. (2026).

## Data Setup

The dataset (recommendations.csv) is archived on Zenodo (comnetsAD/politics_tiktok).
1. Download and extract the dataset archive from Zenodo.
2. Create a local data/ directory inside this project folder.
3. Move recommendations.csv into data/recommendations.csv.

## Running the Script

Prerequisite: Python 3.12

1. Create and activate virtual environment:
   python3.12 -m venv .venv
   source .venv/bin/activate

2. Install dependencies:
   python -m pip install -r requirements.txt

3. Run analysis:
   python tiktok_analysis.py

The script takes ~15-30 seconds to run (including 20,000 block-bootstrap iterations), prints summary results to the terminal, and exports analysis_results/results_full_precision.csv.

## Methodology Notes

- Data Scope: Isolates 22,397 partisan-conditioned account records across Georgia, New York, and Texas.
- 4-Layer Weighting Hierarchy: Sequential averaging: Run Level -> State-Week Cell -> Equal Week Average per State -> Equal 1/3 Overall Weight.
- Resampling: 20,000-iteration calendar-week block bootstrap to account for weekly temporal clustering.

## AI Assistance

Initial code scaffolding and structural logic were drafted with assistance from AI coding tools. All data preprocessing, column mapping, environment execution, and result verification were completed independently by Tim Laux.
