#!/usr/bin/env Rscript

# Load required libraries
library(dplyr)
library(purrr)

# Check for command line arguments
args <- commandArgs(trailingOnly = TRUE)
if(length(args) < 2){
  stop("Usage: Rscript program.R <percentage> <seed>")
}

# Read arguments
percentage <- args[1]
seed <- as.numeric(args[2])
set.seed(seed)

# Define directories
output_dir <- paste0("./Train_", percentage)
training_dir <- file.path(output_dir, "training_datasets")
validation_dir <- file.path(output_dir, "validation_datasets")
test_dir <- file.path(output_dir, "test_datasets")
dirs <- c(training_dir, validation_dir, test_dir)
lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE)

# Read dataset
D <- read.csv("./D.csv")
cat("Total rows in D.csv:", nrow(D), "\n")

# Normalize actions
D$action <- tolower(D$action)
D$action <- gsub("change_to_right", "change_to_right", D$action)
D$action <- gsub("change_to_left", "change_to_left", D$action)
D$action <- gsub("cruise", "cruise", D$action)
D$action <- gsub("keep", "keep", D$action)

# Create expanded dataset with prime variables (for training)
cat("Creating expanded dataset with prime variables...\n")
D_ext <- D %>%
  mutate(curr_lane_prime = lead(curr_lane),
         free_E_prime = lead(free_E),   
         free_NE_prime = lead(free_NE), 
         free_NW_prime = lead(free_NW),
         free_SE_prime = lead(free_SE),
         free_SW_prime = lead(free_SW),
         free_W_prime = lead(free_W)) %>%
  filter(complete.cases(.))

cat("Rows in expanded dataset (D_ext):", nrow(D_ext), "\n")

# Columns used to define uniqueness (original state only)
key_cols <- c("action","curr_lane","free_E","free_NE","free_NW","free_SE","free_SW","free_W")

# Construct keys on both datasets (for internal use only)
D$.key <- do.call(paste, c(D[key_cols], sep="_"))
D_ext$.key <- do.call(paste, c(D_ext[key_cols], sep="_"))

# Create D_unique by keys (this is what we'll split)
D_unique <- D %>% 
  distinct(.key, .keep_all = TRUE) %>%
  select(all_of(key_cols), .key)

cat("Number of unique keys (D_unique):", nrow(D_unique), "\n")

# Create key-to-row mapping for the FULL ORIGINAL dataset
cat("Creating key-to-row mapping...\n")
key_to_rows <- split(1:nrow(D), D$.key)

# Create key-to-row mapping for the EXPANDED dataset
key_to_rows_ext <- split(1:nrow(D_ext), D_ext$.key)

# Stratified 5-fold splitting by action on UNIQUE keys
cat("Performing stratified 5-fold splitting...\n")
folds <- D_unique %>%
  group_by(action) %>%
  group_split() %>%
  map(~ {
    .x <- .x[sample(nrow(.x)), ]  # shuffle rows
    split(.x, rep(1:5, length.out = nrow(.x)))  # split into 5 folds
  })

# Combine folds across actions
folds_combined <- vector("list", 5)
for(i in 1:5){
  folds_combined[[i]] <- bind_rows(lapply(folds, function(f) f[[i]]))
}

# Generate all combinations of 3 training folds, 1 tuning, 1 test
comb <- combn(5, 3)
combinations <- list()
for(j in 1:ncol(comb)){
  train_idx <- comb[,j]
  val_test_idx <- setdiff(1:5, train_idx)
  combinations[[length(combinations)+1]] <- list(train=train_idx, tune=val_test_idx[1], test=val_test_idx[2])
  combinations[[length(combinations)+1]] <- list(train=train_idx, tune=val_test_idx[2], test=val_test_idx[1])
}

# Initialize summary table
summary_list <- list()

# Loop through 20 combinations
cat("Processing fold combinations...\n")
for(i in seq_along(combinations)){
  idx <- combinations[[i]]
  
  # Get keys for each set from the UNIQUE keys split
  train_keys <- bind_rows(folds_combined[idx$train])$.key
  tune_keys <- folds_combined[[idx$tune]]$.key
  test_keys <- folds_combined[[idx$test]]$.key
  
  cat(sprintf("Fold %d: train_keys=%d, tune_keys=%d, test_keys=%d\n", 
              i, length(train_keys), length(tune_keys), length(test_keys)))
  
  # Get row indices for each set using the pre-computed mappings
  train_rows <- unlist(key_to_rows_ext[train_keys])  # From expanded dataset
  tune_rows <- unlist(key_to_rows[tune_keys])        # From original dataset
  test_rows <- unlist(key_to_rows[test_keys])        # From original dataset
  
  cat(sprintf("Fold %d: train_rows=%d, tune_rows=%d, test_rows=%d\n", 
              i, length(train_rows), length(tune_rows), length(test_rows)))
  
  # Apply percentage to training set
  training_percentage <- as.numeric(percentage)
  if(training_percentage < 100){
    n_train <- ceiling(length(train_rows) * (training_percentage / 100))
    train_rows <- sample(train_rows, n_train)
  }
  
  # Create subsets from appropriate datasets
  D_train <- D_ext[train_rows, ]  # From expanded dataset (with prime variables)
  D_tune <- D[tune_rows, ]        # From original dataset
  D_test <- D[test_rows, ]        # From original dataset
  
  # Remove .key column before saving
  D_train <- D_train %>% select(-.key)
  D_tune <- D_tune %>% select(-.key)
  D_test <- D_test %>% select(-.key)
  
  # Save datasets
  write.csv(D_train, file=file.path(training_dir, paste0("train_fold_",i,".csv")), row.names=FALSE)
  write.csv(D_tune,  file=file.path(validation_dir, paste0("validation_fold_",i,".csv")), row.names=FALSE)
  write.csv(D_test,  file=file.path(test_dir, paste0("test_fold_",i,".csv")), row.names=FALSE)
  
  # Overlap counting - check ALL pairs to ensure no leakage
  overlap_train_tune_keys <- length(intersect(train_keys, tune_keys))
  overlap_train_test_keys <- length(intersect(train_keys, test_keys))
  overlap_tune_test_keys <- length(intersect(tune_keys, test_keys))
  
  cat(sprintf("Fold %d: train=%d, tune=%d, test=%d\n", i, nrow(D_train), nrow(D_tune), nrow(D_test)))
  cat(sprintf("   Overlap (keys) train∩tune=%d, train∩test=%d, tune∩test=%d\n", 
              overlap_train_tune_keys, overlap_train_test_keys, overlap_tune_test_keys))
  
  # Verify no leakage between any sets
  stopifnot(overlap_train_tune_keys == 0)
  stopifnot(overlap_train_test_keys == 0)
  stopifnot(overlap_tune_test_keys == 0)
  
  # For summary stats, we need to temporarily recreate keys
  D_train_keys <- do.call(paste, c(D_train[key_cols], sep="_"))
  D_tune_keys <- do.call(paste, c(D_tune[key_cols], sep="_"))
  D_test_keys <- do.call(paste, c(D_test[key_cols], sep="_"))
  
  # Collect summary info
  summary_list[[i]] <- data.frame(
    fold = i,
    train_rows = nrow(D_train),
    tune_rows = nrow(D_tune),
    test_rows = nrow(D_test),
    train_unique_keys = length(unique(D_train_keys)),
    tune_unique_keys = length(unique(D_tune_keys)),
    test_unique_keys = length(unique(D_test_keys)),
    train_dup_rate = 1 - length(unique(D_train_keys))/nrow(D_train),
    tune_dup_rate  = 1 - length(unique(D_tune_keys))/nrow(D_tune),
    test_dup_rate  = 1 - length(unique(D_test_keys))/nrow(D_test),
    overlap_train_tune_keys = overlap_train_tune_keys,
    overlap_train_test_keys = overlap_train_test_keys,
    overlap_tune_test_keys = overlap_tune_test_keys
  )
  
  # Free memory immediately
  rm(D_train, D_tune, D_test, train_rows, tune_rows, test_rows, D_train_keys, D_tune_keys, D_test_keys)
  gc()
}

cat("\nAll splits generated successfully.\n")

summary_df <- bind_rows(summary_list)
write.csv(summary_df, file=file.path(output_dir, "summary.csv"), row.names=FALSE)
cat("\nSummary CSV saved to:", file.path(output_dir, "summary.csv"), "\n")

# Free remaining memory at the end
rm(D, D_ext, key_to_rows, key_to_rows_ext, folds_combined, folds, D_unique, summary_list)
gc()
