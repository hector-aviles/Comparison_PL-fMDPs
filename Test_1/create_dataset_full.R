#!/usr/bin/env Rscript

# -------------------------------------------------
# Load required libraries
# -------------------------------------------------
library(dplyr)

# -------------------------------------------------
# Command line arguments
# -------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript program_single_train.R <seed>")
}

seed <- as.numeric(args[1])
set.seed(seed)

# -------------------------------------------------
# Output directory
# -------------------------------------------------
output_dir <- "./Train_full"
training_dir <- file.path(output_dir, "training_datasets")
dir.create(training_dir, recursive = TRUE, showWarnings = FALSE)

# -------------------------------------------------
# Read dataset
# -------------------------------------------------
D <- read.csv("./complete_DB_discrete.csv")
cat("Total rows in complete_DB_discrete.csv:", nrow(D), "\n")

names(D)
dim(D)

# -------------------------------------------------
# Subset: keep only latent_collision == FALSE
# -------------------------------------------------
cat("Filtering rows with latent_collision == FALSE...\n")

D <- D %>%
  filter(latent_collision == FALSE | latent_collision == "False")

cat("Rows after filtering latent_collision == FALSE:", nrow(D), "\n")

# -------------------------------------------------
# Normalize actions
# -------------------------------------------------
D$action <- tolower(D$action)
D$action <- gsub("change_to_right", "change_to_right", D$action)
D$action <- gsub("change_to_left",  "change_to_left",  D$action)
D$action <- gsub("cruise",           "cruise",           D$action)
D$action <- gsub("keep",             "keep",             D$action)
D$action <- gsub("swerve_left",      "swerve_left",      D$action)
D$action <- gsub("swerve_right",     "swerve_right",     D$action)

# -------------------------------------------------
# Create expanded dataset with prime variables
# -------------------------------------------------
cat("Creating expanded dataset with prime variables...\n")

D_ext <- D %>%
  mutate(
    curr_lane_prime = lead(curr_lane),
    free_E_prime  = lead(free_E),
    free_NE_prime = lead(free_NE),
    free_NW_prime = lead(free_NW),
    free_SE_prime = lead(free_SE),
    free_SW_prime = lead(free_SW),
    free_W_prime  = lead(free_W)
  ) %>%
  filter(complete.cases(.))

cat("Rows in expanded dataset (D_ext):", nrow(D_ext), "\n")

# -------------------------------------------------
# (Optional) shuffle rows for robustness
# -------------------------------------------------
D_ext <- D_ext %>% sample_frac(1.0)

# -------------------------------------------------
# Save single training dataset
# -------------------------------------------------
train_file <- file.path(training_dir, "train_full.csv")
write.csv(D_ext, train_file, row.names = FALSE)

cat("Training dataset saved to:", train_file, "\n")

# -------------------------------------------------
# Summary info
# -------------------------------------------------
key_cols <- c(
  "action", "curr_lane",
  "free_E", "free_NE", "free_NW",
  "free_SE", "free_SW", "free_W"
)

train_keys <- do.call(paste, c(D_ext[key_cols], sep = "_"))

summary_df <- data.frame(
  rows = nrow(D_ext),
  unique_keys = length(unique(train_keys)),
  duplication_rate = 1 - length(unique(train_keys)) / nrow(D_ext)
)

write.csv(
  summary_df,
  file = file.path(output_dir, "summary.csv"),
  row.names = FALSE
)

cat("Summary saved to:", file.path(output_dir, "summary.csv"), "\n")

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
rm(D, D_ext, train_keys)
gc()

