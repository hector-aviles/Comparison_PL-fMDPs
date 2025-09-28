library(dplyr)

# Read the dataset
D_auto <- read.csv("../D.csv")

# Count the total number of unique samples
unique_samples_total <- D_auto %>%
  distinct() %>%
  nrow()

# Count the number of unique samples for each action
unique_samples_per_action <- D_auto %>%
  group_by(action) %>%
  summarise(unique_samples = n_distinct(across(everything())), .groups = 'drop')

# Print the results
cat("Total number of unique samples:", unique_samples_total, "\n")
print("Number of unique samples per action:")
print(unique_samples_per_action)

# If you want to save the results to a CSV file
write.csv(unique_samples_per_action, "unique_samples_per_action.csv", row.names = FALSE)
