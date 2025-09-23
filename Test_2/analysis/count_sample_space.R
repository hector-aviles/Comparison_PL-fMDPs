library(dplyr)

# Read the datasets
css <- read.csv("complete_sample_space.csv")
D_humans <- read.csv("../D_humans.csv")

D_humans <- subset(D_humans, select=-c(driver))

# Add a temporary key to both data frames for counting
css <- css %>%
  mutate(key = do.call(paste, c(across(everything()), sep = "|")))

D_humans <- D_humans %>%
  mutate(key = do.call(paste, c(across(everything()), sep = "|")))

# Count occurrences of each row in css within D_humans
counted_css <- css %>%
  left_join(D_humans %>% group_by(key) %>% summarise(count = n(), .groups = 'drop'), by = "key") %>%
  mutate(count = ifelse(is.na(count), 0, count)) %>%
  select(-key)  # Remove the temporary key column

# Add index column starting from 1
counted_css <- counted_css %>%
  mutate(index = row_number())

# Write the result to a CSV file
write.csv(counted_css, "count_sample_space.csv", row.names = FALSE)

