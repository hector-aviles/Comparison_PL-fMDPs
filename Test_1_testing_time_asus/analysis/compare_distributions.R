# Load necessary libraries
library(ggplot2)
library(dplyr)
#install.packages("philentropy")
#install.packages("DescTools")
library(philentropy)
library(DescTools)
library(transport)
#chooseCRANmirror()  # Select a mirror closer to your location
#install.packages("transport")

# Read CSV files into dataframes
df_auto <- read.csv("count_sample_space_auto.csv")
df_humans <- read.csv("count_sample_space_humans.csv")

# Ensure both dataframes contain 'index' and 'count' columns
if (!all(c("index", "count") %in% colnames(df_auto)) || !all(c("index", "count") %in% colnames(df_humans))) {
  stop("Both dataframes must contain 'index' and 'count' columns.")
}

# Compute cumulative frequencies
df_auto <- df_auto %>%
  arrange(index) %>%
  mutate(cumulative_count = cumsum(count))

df_humans <- df_humans %>%
  arrange(index) %>%
  mutate(cumulative_count = cumsum(count))

# Normalize cumulative frequencies to compare them properly
df_auto$cumulative_freq <- df_auto$cumulative_count
df_humans$cumulative_freq <- df_humans$cumulative_count

# Merge for plotting cumulative frequencies
df_combined <- merge(df_auto[, c("index", "cumulative_freq")], 
                     df_humans[, c("index", "cumulative_freq")], 
                     by = "index", suffixes = c("_auto", "_humans"))

# Plot cumulative frequency functions
p1 <- ggplot(df_combined, aes(x = index)) +
  geom_line(aes(y = cumulative_freq_auto, color = "Autonomous"), linewidth = 1) +  # Line for autonomous
  geom_point(aes(y = cumulative_freq_auto, color = "Autonomous"), size = 3, shape = 21, fill = "white", stroke = 1) +  # Circles for autonomous
  geom_line(aes(y = cumulative_freq_humans, color = "Human"), linewidth = 1) +  # Line for human
  geom_point(aes(y = cumulative_freq_humans, color = "Human"), size = 3, shape = 21, fill = "white", stroke = 1) +  # Circles for human
  scale_color_manual(values = c("Autonomous" = "green", "Human" = "red")) +  # Custom colors
  labs(
    x = "State-action examples",
    y = "Cumulative frequency",
    color = "Control type"
  ) +
  theme_bw() +  # Add frame around the plot
  theme(
    aspect.ratio = 1,  # Square aspect ratio
    legend.position = c(.2, .9),  # Position legend
    axis.text.x = element_text(size = 12),  # Adjust x-axis text size
    axis.text.y = element_text(size = 12),  # Adjust y-axis text size
    axis.title.x = element_text(size = 18),  # Adjust x-axis title size
    axis.title.y = element_text(size = 18),  # Adjust y-axis title size
    plot.title = element_text(size = 20),  # Adjust plot title size
    legend.text = element_text(size = 14),  # Increase legend text size
    legend.title = element_text(size = 16)  # Increase legend title size
  )
# Merge for histogram comparison
df_auto$control_type <- "Autonomous"
df_humans$control_type <- "Human"
df_hist <- rbind(df_auto[, c("index", "count", "control_type")], 
                 df_humans[, c("index", "count", "control_type")])

# Plot superimposed histogram
p2 <- ggplot(df_hist, aes(x = index, y = count, fill = control_type)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
  scale_fill_manual(values = c("Autonomous" = "green", "Human" = "red")) +
  labs(title = "",
       x = "State-action examples",
       y = "Frequency",
       fill = "Control Type") +
  theme_bw() + 
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 18),
    plot.title = element_text(size = 20),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 16)
  )

# Display the plots
ggsave("cumulative_distributions.pdf", plot = p1, dpi = 300)
ggsave("superimposed_histograms.pdf", plot = p2, dpi = 300)


# Perform Kolmogorov-Smirnov test
ks_test_result <- ks.test(df_auto$cumulative_freq, df_humans$cumulative_freq)
print("Kolmogorov-Smirnov Test Results:")
print(ks_test_result)

# Perform Chi-Square test
# Ensure frequencies are aligned by index
merged_counts <- merge(df_auto[, c("index", "count")], df_humans[, c("index", "count")], by = "index", suffixes = c("_auto", "_humans"))
chisq_test_result <- chisq.test(merged_counts$count_auto, merged_counts$count_humans)

print("Chi-Square Test Results:")
print(chisq_test_result)



# Normalize the counts to obtain probability distributions
df_auto <- df_auto %>%
  arrange(index) %>%
  mutate(prob = count / sum(count))

df_humans <- df_humans %>%
  arrange(index) %>%
  mutate(prob = count / sum(count))

# Merge probabilities for visualization and KL divergence calculation
df_combined <- merge(df_auto[, c("index", "prob")], 
                     df_humans[, c("index", "prob")], 
                     by = "index", suffixes = c("_auto", "_humans"))

# Avoid zero probabilities for KL divergence by replacing zeros with small values
df_combined$prob_auto[df_combined$prob_auto == 0] <- 1e-10
df_combined$prob_humans[df_combined$prob_humans == 0] <- 1e-10

# Compute KL divergence per index
df_combined$KL_div <- df_combined$prob_auto * log(df_combined$prob_auto / df_combined$prob_humans) + 
                      df_combined$prob_humans * log(df_combined$prob_humans / df_combined$prob_auto)

# Compute Jensen-Shannon Divergence
prob_matrix <- rbind(df_combined$prob_auto, df_combined$prob_humans)
jsd_result <- JSD(prob_matrix, unit = "log2")

print(paste("Jensen-Shannon Divergence:", jsd_result))


# Plot KL divergence per index
p3 <- ggplot(df_combined, aes(x = index, y = KL_div)) +
  geom_bar(stat = "identity", fill = "purple", alpha = 0.6) +
  labs(title = "Kullback-Leibler Divergence per Index",
       x = "State-action examples",
       y = "KL Divergence") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 18),
    plot.title = element_text(size = 20)
  )

# Plot probability distributions (overlayed)
p4 <- ggplot(df_combined, aes(x = index)) +
  geom_line(aes(y = prob_auto, color = "Autonomous"), linewidth = 1) +
  geom_line(aes(y = prob_humans, color = "Human"), linewidth = 1, linetype = "dashed") +
  scale_color_manual(values = c("Autonomous" = "green", "Human" = "red")) +
  labs(title = "",
       x = "State-action examples",
       y = "Relative frequency",
       color = "Control type") +
  theme_bw() + 
  theme(
    legend.position=c(.2,.9),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 18),
    plot.title = element_text(size = 20),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 16)
  )

# Display plots
# Display the plots
ggsave("KL_divergence.pdf", plot = p3, dpi = 300)
ggsave("prob_distributions_overlayed.pdf", plot = p4, dpi = 300)

############################################################################

# Read CSV files
df_auto <- read.csv("count_sample_space_auto.csv")
df_humans <- read.csv("count_sample_space_humans.csv")

# Ensure both dataframes have 'index' and 'count' columns
if (!all(c("index", "count") %in% colnames(df_auto)) || !all(c("index", "count") %in% colnames(df_humans))) {
  stop("Both dataframes must contain 'index' and 'count' columns.")
}

# Merge datasets by index to ensure aligned comparison
merged_counts <- merge(df_auto[, c("index", "count")], df_humans[, c("index", "count")], 
                       by = "index", suffixes = c("_auto", "_humans"))

# Normalize counts to get probability distributions
merged_counts$prob_auto <- merged_counts$count_auto / sum(merged_counts$count_auto)
merged_counts$prob_humans <- merged_counts$count_humans / sum(merged_counts$count_humans)

# Compute Wasserstein Distance (Earth Mover's Distance)
wasserstein_dist <- wasserstein1d(merged_counts$prob_auto, merged_counts$prob_humans)
print(paste("Wasserstein Distance:", wasserstein_dist))

# Compute Jensen-Shannon Divergence
prob_matrix <- rbind(merged_counts$prob_auto, merged_counts$prob_humans)
jsd_value <- JSD(prob_matrix, unit = "log2")
print(paste("Jensen-Shannon Divergence:", jsd_value))

# ---------------------
# Permutation Test for JSD
# ---------------------

set.seed(123)  # Set seed for reproducibility
num_permutations <- 1000  # Number of shuffles

# Function to compute JSD for shuffled distributions
permute_jsd <- function() {
  shuffled <- sample(merged_counts$prob_auto)  # Shuffle auto probabilities
  prob_matrix_perm <- rbind(shuffled, merged_counts$prob_humans)
  return(JSD(prob_matrix_perm, unit = "log2"))
}

# Generate permuted JSD values
jsd_permutations <- replicate(num_permutations, permute_jsd())

# Compute p-value: proportion of permuted JSD values greater than observed JSD
p_value <- mean(jsd_permutations >= jsd_value)

# Print permutation test results
print(paste("Permutation Test p-value for JSD:", p_value))

# Optional: Plot histogram of permuted JSD values
#hist(jsd_permutations, breaks = 30, col = "lightblue", main = "Permutation Test for JSD",
#     xlab = "JSD Values", ylab = "Frequency")
#abline(v = jsd_value, col = "red", lwd = 2, lty = 2)  # Observed JSD line
#legend("topright", legend = "Observed JSD", col = "red", lwd = 2, lty = 2) but transport package cannot be installed neither with R command line nor with RStudio
