library(ggplot2)
library(dplyr)
library(philentropy)
library(DescTools)
library(transport)

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

# Normalize cumulative frequencies to 1
df_auto$cumulative_freq <- df_auto$cumulative_count / max(df_auto$cumulative_count)
df_humans$cumulative_freq <- df_humans$cumulative_count / max(df_humans$cumulative_count)

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
    y = "Normalized cumulative frequency",
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



# Display the plots
ggsave("cumulative_distributions_v2.pdf", plot = p1, dpi = 300)

