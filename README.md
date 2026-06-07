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
- `data/processed/breed_anaylsis_q1.csv`: Question 1 modeling data in the modified CSV format used by the course Random Forest implementation.

## Question 1 Modeling Data

The Question 1 processed file follows the course's mixed-data CSV format:

1. Column names.
2. Domain/type codes.
3. Class variable name.
4. Data rows.

For `data/processed/breed_anaylsis_q1.csv`, the type-code row means:

- `-1`: row identifier, ignored by the classifier (`Breed`).
- `0`: numeric predictor.
- positive integer: categorical predictor or class variable with that many possible values.

The class variable is `Popularity Tier`.