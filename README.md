# CSC466 Final Project

Summer Mariana Ortega (sorteg16@calpoly.edu)  
Diego Melgoza (drmelgoz@calpoly.edu)

## Overview

This project investigates the logic behind dog breed popularity in the United States using American Kennel Club (AKC) breed traits, breed groups, and popularity rankings from 2013-2025.

## Research Questions

1. Can dog breed traits predict a breed's popularity tier in the United States?
   - Model approach: Random Forest classification
   - Target variable: `Popularity Tier`

2. Do trait-based groups of dog breeds differ in popularity, and how do those groups compare to AKC breed groups?
   - Model approach: Clustering

## Data Files

- `data/raw/raw_dog_traits_table_2013_2025.csv`: combined AKC source data before project-specific cleaning.
- `data/interim/breed_traits.csv`: breed trait table.
- `data/interim/breed_ranks.csv`: breed popularity rankings from 2013-2025.
- `data/interim/breed_groups.csv`: AKC breed group labels.
- 
- `data/processed/breed_traits_full.csv`: Question 1 modeling data with all individual trait ratings.
- `data/processed/breed_traits_mean.csv`: Question 1 modeling data with AKC trait-group mean ratings.

## Repository Structure

- `configs/`: reusable model configuration files.
- `data/raw/`: original combined source data.
- `data/interim/`: cleaned intermediate data tables used to build modeling datasets.
- `data/processed/`: final modeling datasets only.
- `models/`: reusable Python model code from the CSC 466 labs.
- `notebooks/`: project notebooks for cleaning, setup, modeling, and analysis.

## Question 1 Modeling Data

For `data/processed/breed_traits_full.csv` and `data/processed/breed_traits_mean.csv`, the type-code row means:

- `-1`: row identifier, ignored by the classifier (`Breed`).
- `0`: numeric predictor.
- positive integer: categorical predictor or class variable with that many possible values.

The class variable is `Popularity Tier`, currently defined as three average-rank groups:

- `High Popularity`: average rank 1-50.
- `Medium Popularity`: average rank 51-125.
- `Low Popularity`: average rank 126+.

## Question 1 Model Files

- `models/randomForest/gridSearch.py`: runs cross-validation over the custom Random Forest grid and returns the best parameters as a dictionary.
- `notebooks/random_forest.ipynb`: evaluates the custom Random Forest model using all individual trait ratings.
- `notebooks/random_forest_mean.ipynb`: evaluates the custom Random Forest model using trait-group mean ratings.
