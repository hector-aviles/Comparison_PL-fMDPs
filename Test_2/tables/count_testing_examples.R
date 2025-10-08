library(tidyverse)
library(xtable)

# Define the percentages and base directory structure
percentages <- c("01", "50", "100")
base_dirs <- paste0("../Test_", percentages, "/test_datasets/")

# Initialize results dataframe
results <- data.frame(
  Percentage = character(),
  Max_Examples = integer(),
  Min_Examples = integer(),
  stringsAsFactors = FALSE
)

# Function to count lines in CSV (excluding header)
count_examples <- function(file_path) {
  if (file.exists(file_path)) {
    # Read the file and count rows (excluding header)
    data <- read.csv(file_path)
    return(nrow(data))
  } else {
    warning(paste("File not found:", file_path))
    return(NA)
  }
}

# Process each percentage directory
for (i in seq_along(percentages)) {
  percentage <- percentages[i]
  base_dir <- base_dirs[i]
  
  # Initialize vectors to store counts for all folds
  example_counts <- numeric(10)
  
  # Count examples for each fold (1 to 10)
  for (fold in 1:10) {
    file_path <- paste0(base_dir, "test_fold_", fold, ".csv")
    example_counts[fold] <- count_examples(file_path)
  }
  
  # Remove any NA values (in case files were missing)
  example_counts <- example_counts[!is.na(example_counts)]
  
  if (length(example_counts) > 0) {
    # Calculate max and min
    max_examples <- max(example_counts)
    min_examples <- min(example_counts)
    
    # Add to results
    results <- rbind(results, data.frame(
      Percentage = paste0(percentage, "%"),
      Max_Examples = max_examples,
      Min_Examples = min_examples
    ))
  }
}

# Format numbers with commas for better readability in LaTeX
results$Max_Examples_Formatted <- format(results$Max_Examples, big.mark = ",", scientific = FALSE)
results$Min_Examples_Formatted <- format(results$Min_Examples, big.mark = ",", scientific = FALSE)

# Create LaTeX table
latex_table <- paste(
  "\\begin{table}[ht!]",
  "\\centering",
  " \\caption{Range of human control examples used for testing models across 10 folds, as a function of the testing data percentage from $\\mathcal{D}_{humans}$.}",
  " \\label{tab:size_test_2}",
  " %\\resizebox{0.75\\textwidth}{!}{%",
  "%\\small",
  "\\begin{tabular}{lcc}",
  "\\toprule",
  " &  \\multicolumn{2}{c}{Number of examples from $\\mathcal{D}_{humans}$}  \\\\",
  "\\cmidrule(lr){2-3}",
  "Data \\%  & Maximum & Minimum \\\\",
  "\\midrule",
  paste0(results$Percentage[1], " &  ", results$Max_Examples_Formatted[1], " & ", results$Min_Examples_Formatted[1], "  \\\\"),
  paste0(results$Percentage[2], " &  ", results$Max_Examples_Formatted[2], " & ", results$Min_Examples_Formatted[2], " \\\\"),
  paste0(results$Percentage[3], " &  ", results$Max_Examples_Formatted[3], " & ", results$Min_Examples_Formatted[3], " \\\\"),
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}",
  sep = "\n"
)

# Print the LaTeX table
cat(latex_table)

# Also print the results in a readable format for verification
print("Summary of results:")
print(results[, c("Percentage", "Max_Examples", "Min_Examples")])

# Write the LaTeX table to a file (optional)
writeLines(latex_table, "test_size_table.tex")
