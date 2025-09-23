# Eliminar columnas, renombrar acciones 

library(dplyr)
library(stringr)
library(tidyr)
# There are conflicts between tidyverse an dplyr
# https://stackoverflow.com/questions/73336628/package-conflicts-in-r
#install.packages("conflicted")
library(conflicted)  
#install.packages("tidyverse")
library(tidyverse)
conflict_prefer("filter", "dplyr")
conflict_prefer("lag", "dplyr")
# Required to use discretization function
#install.packages("arules")
library("arules")


df <- read.table("../D_humans.csv", header=T, sep = ",")

diff <- df %>%
  group_by(action,  curr_lane , free_E , free_NE , free_NW , free_SE , free_SW,free_W) %>%
  summarize(count = n())

View(diff)

df_ordered <- diff %>%
  arrange(desc(curr_lane), desc(count))

View(df_ordered)

write.csv(df_ordered, "D_frequencies.csv", row.names = FALSE)



