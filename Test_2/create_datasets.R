#!/usr/bin/env Rscript

# Load required libraries
library(dplyr)
library(tidyr)
library(purrr)

set.seed(123)

# Check for command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript program.R <percentage>")
}

# Read the percentage from command line
percentage <- args[1]

# Ensure output directories exist
output_dir <- paste0("./Test_", percentage, "/test_datasets/")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Read the dataset
D <- read.csv("./D_humans.csv")
dim(D)

# Stratified sampling
sampling_percentage <- as.numeric(percentage) / 100
print(paste0("Sampling percentage: ", sampling_percentage))
stratified_sample <- D %>%
  group_by(action, driver) %>%
  sample_frac(sampling_percentage)
dim(stratified_sample)

# Write the sampled dataset
test_path <- paste0("./Test_", percentage, "/D_humans_", percentage, ".csv")
write.csv(stratified_sample, test_path, row.names = FALSE)

# Divide the stratified_sample into 20 equal parts, stratified by action and driver
num_subsets <- 20
# Ensure disjoint subsets with equal distribution
set.seed(123)  # Set a seed for reproducibility
stratified_subsets <- stratified_sample %>%
  group_by(action, driver) %>%
  mutate(subset_id = sample(rep(1:num_subsets, length.out = n()))) %>%
  ungroup()

# Save each subset as a separate CSV file
for (i in 1:num_subsets) {
  subset_data <- stratified_subsets %>%
    filter(subset_id == i) %>%
    select(-subset_id)  # Remove the subset_id column before saving
  
  subset_path <- paste0(output_dir, "/test_fold_", i, ".csv")
  write.csv(subset_data, subset_path, row.names = FALSE)
  
  
  size_dir <- paste0("./Test_", percentage)
  # Record size for the first repetition
  if (i == 1) {
    size_df <- data.frame(i = i, sample_size= nrow(stratified_sample), test_size=nrow(subset_data))
    write.csv(size_df, file = file.path(size_dir, "size.csv"), row.names = FALSE)
  } else {
    size_df <- data.frame(i = i, sample_size= nrow(stratified_sample), test_size=nrow(subset_data))
    write.table(size_df,
                file = file.path(size_dir, "size.csv"),
                append = TRUE,
                sep = ",",
                col.names = FALSE,
                row.names = FALSE)
  }  
  
}

print("All subsets saved successfully!")
