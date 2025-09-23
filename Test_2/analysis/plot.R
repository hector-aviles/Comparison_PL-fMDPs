library(ggplot2)

# Read the CSV file (modify the filename as needed)
df <- read.csv("count_sample_space.csv")

# Ensure the dataframe has the required columns
if (!all(c("index", "count") %in% colnames(df))) {
  stop("The dataframe must contain 'index' and 'count' columns.")
}

# Plot the histogram with a border (frame)
p <- ggplot(df, aes(x = index, y = count)) +
  geom_bar(stat = "identity", fill = "green", color = "green") +
  scale_y_continuous(limits = c(0, 45000), breaks = seq(0, 45000, by = 5000)) +
  labs(title = "",
       x = "State-action examples",
       y = "Number of repetitions") +
  # Fix y-axis range
  theme_bw() +  # Add frame around the plot
  theme(
    axis.text.x = element_text(size = 12),  # Bigger x-axis text
    axis.text.y = element_text(size = 12),  # Bigger y-axis text
    axis.title.x = element_text(size = 22),  # Bigger x-axis title
    axis.title.y = element_text(size = 22),  # Bigger y-axis title
    plot.title = element_text(size = 30)  # Bigger plot title
  )

# Display the plot
#print(p)


ggsave("D_humans_unique_count.pdf", plot = p, dpi = 300)

