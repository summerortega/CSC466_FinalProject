# CSC466 Final Project

Summer Mariana Ortega (sorteg16@calpoly.edu)  
Diego Melgoza (drmelgoz@calpoly.edu)

## Overview

This project investigates the logic behind dog breed popularity in the United States using American Kennel Club (AKC) breed traits, breed groups, and popularity rankings from 2013-2025.

## Research Questions

1. Can dog breed traits predict a breed's popularity tier in the United States?
   - Model approach: Random Forest classification
   - Target variable: `Popularity Tier`

2. Can dog breeds be grouped into meaningful clusters based on their traits, and do those trait-based clusters differ in popularity or align with official AKC breed groups?
   - Model approach: Clustering
   - Comparison labels: `Popularity Tier` and `AKC Group`

## Data Files

- `data/raw/raw_dog_traits_table_2013_2025.csv`: combined AKC source data before project-specific cleaning.
- `data/interim/breed_traits.csv`: breed trait table.
- `data/interim/breed_ranks.csv`: breed popularity rankings from 2013-2025.
- `data/interim/breed_groups.csv`: AKC breed group labels.
- `data/processed/breed_forest_full.csv`: Question 1 modeling data with all individual trait ratings.
- `data/processed/breed_forest_avg.csv`: Question 1 modeling data with AKC trait-group average ratings.
- `data/processed/breed_clustering_full.csv`: Question 2 clustering data with all individual trait ratings plus comparison labels.
- `data/processed/breed_clustering_avg.csv`: Question 2 clustering data with AKC trait-group average ratings plus comparison labels.

## Repository Structure

- `data/raw/`: original combined source data.
- `data/interim/`: cleaned intermediate data tables used to build modeling datasets.
- `data/processed/`: final modeling datasets only.
- `models/`: models from CSC 466 labs.
- `notebooks/`: project notebooks for cleaning, setup, modeling, and analysis.
- `results/`: model outputs.

## Question 1 Modeling Data

For `data/processed/breed_forest_full.csv` and `data/processed/breed_forest_avg.csv`, the type-code row means:

- `-1`: row identifier, ignored by the classifier (`Breed`).
- `0`: numeric predictor.
- positive integer: categorical predictor or class variable with that many possible values.

The class variable is `Popularity Tier`, currently defined as three average-rank groups:

- `High Popularity`: average rank 1-50.
- `Medium Popularity`: average rank 51-125.
- `Low Popularity`: average rank 126+.

## Question 1 Model Files

- `notebooks/random_forest_full.ipynb`: evaluates the custom Random Forest model using all individual trait ratings.
- `notebooks/random_forest_avg.ipynb`: evaluates the custom Random Forest model using trait-group average ratings.
- `models/randomForest/rf_analysis_utils.py`: shared Random Forest notebook helpers for train/test loading, interpretation tables, confusion matrices, prediction CSVs, and rank-based summaries.

## Question 2 Clustering Files

- `notebooks/model2_setup.ipynb`: creates the Question 2 clustering dataset.

For `data/processed/breed_clustering_full.csv` and `data/processed/breed_clustering_avg.csv`, the second row marks the clustering columns:

- `1`: numeric column used by the clustering algorithm.
- `0`: metadata or ground-truth label column not used as a clustering feature.
