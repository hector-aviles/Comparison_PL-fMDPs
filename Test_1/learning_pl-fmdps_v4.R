library(bnlearn)
library(dplyr)
library(stringr)
library(tidyr)
library(conflicted)
library(tidyverse)
conflict_prefer("filter", "dplyr")
conflict_prefer("lag", "dplyr")
library(arules)

get_BN <- function(bn_fit, action_str, output) {
   file <- file(output, "a")
   writeLines("%%%%%%%%", con = file)  
   action_label <- paste("% ", action_str)
   writeLines(action_label, con = file)  
   writeLines("%%%%%%%%\n", con = file)  
   
   for (rv in bn_fit) {
     rv_name <- rv$node
     rv_values <- dimnames(rv$prob)[[1]]   
     rv_numval = length(rv_values) 
          
     if (!identical(rv$parents, character(0))){ 
         print("Processing RV: ")
         print(rv_name)
         df2 <- as.data.frame(rv$prob)
         df2[is.na(df2)] <- 0.0
         df2[] <- lapply(df2, as.character)
         df2$Freq <- as.numeric(df2$Freq)
     
         if (rv_numval > 2 | rv_numval == 2){  
           for (i in seq(from = 1, to = nrow(df2), by = rv_numval)){
              count_zeros <- 0
              head <- ""
              for (j in seq(from = i, to = i + rv_numval - 1, by = 1)){
                 if (identical(tolower(rv_values[1]), 'false') & identical(tolower(rv_values[2]), 'true')){
                    name <- gsub("_prime", "", rv_name) 
                    head <- paste(head, df2[j + 1, ]$Freq, "::", name, "(1)", sep = "", collapse = "") 
                    if (df2[j + 1, ]$Freq == 0.0){
                       count_zeros <- rv_numval
                    }              
                    break    
                 }
                 head <- paste(head, df2[j, ]$Freq, "::", name, "(", df2[j,1], ")", sep = "", collapse = "")
                 if (j < i + rv_numval - 1){
                    head <- paste(head, "; ", sep = "", collapse = "")
                 } 
                 if (df2[j, ]$Freq == 0.0) {
                    count_zeros <- count_zeros + 1
                 }   
              }
              if (count_zeros < rv_numval){
                 body <- ""
                 for (k in seq(from = 2, to = ncol(df2) - 1, by = 1)){
                    col_values <- unique(as.character(unlist(df2[, k])))
                    if (length(col_values) > 2){
                       body <- paste(body, names(df2)[k], "(0)","(", df2[i, k], ")", sep = "", collapse = "")      
                    } else {
                       if (identical(tolower(col_values[1]), 'false') & identical(tolower(col_values[2]), 'true')){  
                          if (identical(tolower(df2[i, k]), 'false')){
                             body <- paste(body, "not(", names(df2)[k], "(0)",")", sep = "", collapse = "")
                          } else {
                             body <- paste(body, names(df2)[k], "(0)", sep = "", collapse = "")
                          }
                       } else {
                          body <- paste(body, names(df2)[k], "(0)","(", df2[i, k], ")", sep = "", collapse = "")
                       }
                    } 
                    if (k == ncol(df2) - 1){
                       body <- paste(body, ", ", action_str, ".", sep = "", collapse = "") 
                    } else {
                       body <- paste(body, ", ", sep = "", collapse = "") 
                    }
                 }
                 rule <- paste(head, ":-", body)
                 print(rule)
                 writeLines(rule, con = file)
              }   
           }
         }
     }
   }
   
   for (rv in bn_fit) {
     rv_name <- rv$node
     if (grepl("_prime", rv_name)){
       rv_values <- dimnames(rv$prob)[[1]]   
       rv_numval = length(rv_values) 
       if (identical(rv$parents, character(0))){ 
         print("Processing RV: ")
         print(rv_name)
         df2 <- as.data.frame(rv$prob)
         df2[is.na(df2)] <- 0.0
         df2[] <- lapply(df2, as.character)
         df2$Freq <- as.numeric(df2$Freq)
         if ((length(rv_values) == 2) & identical(tolower(rv_values[1]), 'false') & identical(tolower(rv_values[2]), 'true')){
           name <- gsub("_prime", "", rv_name)
           fact_pos <- paste(df2[2, ]$Freq, "::", name, "(1)", ":- ", name, "(0), ", sep = "", collapse = "")
           fact_neg <- paste((1 - df2[2, ]$Freq), "::", name, "(1)", ":- ", "not(", name, "(0)), ", sep = "", collapse = "")
           fact_neg <- paste(fact_neg, action_str, ".", sep = "", collapse = "")
           print(fact_neg)
           writeLines(fact_neg, con = file)
           fact_pos <- paste(fact_pos, action_str, ".", sep = "", collapse = "")
           print(fact_pos)
           writeLines(fact_pos, con = file)            
         }
       }
     }
   }
   writeLines("\n", con = file)
   close(file)
}

correct_unique <- function(df) {
   if (length(unique(df$free_NE)) == 1) {
     if (df$free_NE[1] == "True"){
        df$free_NE[1] <- "False"
        df$free_NE_prime[1] <- "False"
     } else {
        df$free_NE[1] <- "True"
        df$free_NE_prime[1] <- "True"
     } 
   }
   if (length(unique(df$free_NW)) == 1) {
     if (df$free_NW[1] == "True"){
        df$free_NW[1] <- "False"
        df$free_NW_prime[1] <- "False"
     } else {
        df$free_NW[1] <- "True"
        df$free_NW_prime[1] <- "True"
     } 
   }
   if (length(unique(df$free_E)) == 1) {
     if (df$free_E[1] == "True"){
        df$free_E[1] <- "False"
        df$free_E_prime[1] <- "False"
     } else {
        df$free_E[1] == "True"
        df$free_E_prime[1] <- "True"
     } 
   }
   if (length(unique(df$free_SE)) == 1) {
     if (df$free_SE[1] == "True"){
        df$free_SE[1] <- "False"
        df$free_SE_prime[1] <- "False"
     } else {
        df$free_SE[1] == "True"
        df$free_SE_prime[1] <- "True"
     } 
   }
   if (length(unique(df$free_W)) == 1) {
     if (df$free_W[1] == "True"){
        df$free_W[1] <- "False"
        df$free_W_prime[1] <- "False"
     } else {
        df$free_W[1] == "True"
        df$free_W_prime[1] == "True"
     } 
   }
   if (length(unique(df$free_SW)) == 1) {
     if (df$free_SW[1] == "True"){
        df$free_SW[1] <- "False"
        df$free_SW_prime[1] <- "False"
     } else {
        df$free_SW[1] == "True"
        df$free_SW_prime[1] == "True"
     } 
   }
   if (length(unique(df$curr_lane)) == 1) {
     if (df$curr_lane[1] == "True"){
        df$curr_lane[1] <- "False"
        df$curr_lane_prime[1] <- "False"
     } else {
        df$curr_lane[1] == "True"
        df$curr_lane_prime[1] == "True"
     } 
   }
   return(df)
}

adjust_values <- function(freq) {
   freq[is.na(freq)] <- 0.0
   freq[freq <= 0.0001] <- 0.001
   total_sum = sum(freq)
   if (total_sum == 0.0) {
      freq <- 1 / length(freq)
   } else if (total_sum > 1) {
      freq <- freq / total_sum
   }
   return(freq)
}

pause = function(){
   while (TRUE) {
      input <- readline(prompt="An error occurred: Press CTRL-C key to continue (any subsequent code in the script was ignored)")
   }    
}
options(error=pause)

args <- commandArgs(trailingOnly = TRUE)

datafile <- "./train_fold_1.csv"
output <- "./pl_mdp_1.pl"
gamma <- 0.9
epsilon <- 0.1

if (length(args) >= 2) {
   datafile <- args[1]
   output <- args[2]
}
if (length(args) >= 4) {
   gamma <- as.numeric(args[3])
   epsilon <- as.numeric(args[4])
}

print(paste("datafile:", datafile))
print(paste("output:", output))
print(paste("gamma:", gamma))
print(paste("epsilon:", epsilon))

df <- read.table(datafile, header=TRUE, sep=",")
dim(df)
names(df)

bl_nep <- data.frame(from = c("free_NE_prime", "free_NE_prime", "free_NE_prime", "free_NE_prime", "free_NE_prime", "free_NE_prime", "free_NE_prime", "free_NE_prime", "free_NE_prime", "free_NE_prime", "free_NE_prime", "free_NE_prime", "free_NE_prime"), to = c("free_NE", "free_NW", "free_E", "free_SE", "free_W", "free_SW", "curr_lane", "free_NW_prime", "free_E_prime", "free_SE_prime", "free_W_prime", "free_SW_prime", "curr_lane_prime"))
bl_nwp <- data.frame(from = c("free_NW_prime", "free_NW_prime", "free_NW_prime", "free_NW_prime", "free_NW_prime", "free_NW_prime", "free_NW_prime", "free_NW_prime", "free_NW_prime", "free_NW_prime", "free_NW_prime", "free_NW_prime", "free_NW_prime"), to = c("free_NE", "free_NW", "free_E", "free_SE", "free_W", "free_SW", "curr_lane", "free_NE_prime", "free_E_prime", "free_SE_prime", "free_W_prime", "free_SW_prime", "curr_lane_prime"))
bl_ep <- data.frame(from = c("free_E_prime", "free_E_prime", "free_E_prime", "free_E_prime", "free_E_prime", "free_E_prime", "free_E_prime", "free_E_prime", "free_E_prime", "free_E_prime", "free_E_prime", "free_E_prime", "free_E_prime"), to = c("free_NE", "free_NW", "free_E", "free_SE", "free_W", "free_SW", "curr_lane", "free_NE_prime", "free_NW_prime", "free_SE_prime", "free_W_prime", "free_SW_prime", "curr_lane_prime"))
bl_sep <- data.frame(from = c("free_SE_prime", "free_SE_prime", "free_SE_prime", "free_SE_prime","free_SE_prime", "free_SE_prime", "free_SE_prime", "free_SE_prime", "free_SE_prime", "free_SE_prime", "free_SE_prime", "free_SE_prime", "free_SE_prime"), to = c("free_NE", "free_NW", "free_E", "free_SE", "free_W", "free_SW", "curr_lane", "free_NE_prime", "free_NW_prime", "free_E_prime", "free_W_prime", "free_SW_prime", "curr_lane_prime"))
bl_wp <- data.frame(from = c("free_W_prime", "free_W_prime", "free_W_prime", "free_W_prime", "free_W_prime", "free_W_prime", "free_W_prime", "free_W_prime", "free_W_prime", "free_W_prime", "free_W_prime", "free_W_prime", "free_W_prime"), to = c("free_NE", "free_NW", "free_E", "free_SE", "free_W", "free_SW", "curr_lane", "free_NE_prime", "free_NW_prime", "free_E_prime", "free_SE_prime", "free_SW_prime", "curr_lane_prime"))
bl_swp <- data.frame(from = c("free_SW_prime", "free_SW_prime", "free_SW_prime", "free_SW_prime", "free_SW_prime", "free_SW_prime", "free_SW_prime", "free_SW_prime", "free_SW_prime", "free_SW_prime", "free_SW_prime", "free_SW_prime", "free_SW_prime"), to = c("free_NE", "free_NW", "free_E", "free_SE", "free_W", "free_SW", "curr_lane", "free_NE_prime", "free_NW_prime", "free_E_prime", "free_W_prime", "free_SE_prime", "curr_lane_prime"))
bl_clp <- data.frame(from = c("curr_lane_prime", "curr_lane_prime", "curr_lane_prime", "curr_lane_prime", "curr_lane_prime", "curr_lane_prime", "curr_lane_prime", "curr_lane_prime", "curr_lane_prime", "curr_lane_prime", "curr_lane_prime", "curr_lane_prime", "curr_lane_prime"), to = c("free_NE", "free_NW", "free_E", "free_SE", "free_W", "free_SW", "curr_lane", "free_NE_prime", "free_NW_prime", "free_E_prime", "free_W_prime", "free_SE_prime", "free_SW_prime"))
bl_ne <- data.frame(from = c("free_NE", "free_NE", "free_NE", "free_NE", "free_NE", "free_NE"), to = c("free_NW", "free_E", "free_SE", "free_W", "free_SW", "curr_lane"))
bl_nw <- data.frame(from = c("free_NW", "free_NW", "free_NW", "free_NW", "free_NW", "free_NW"), to = c("free_NE", "free_E", "free_SE", "free_W", "free_SW", "curr_lane"))
bl_e <- data.frame(from = c("free_E", "free_E", "free_E", "free_E", "free_E", "free_E"), to = c("free_NE","free_NW","free_SE", "free_W", "free_SW", "curr_lane"))
bl_se <- data.frame(from = c("free_SE", "free_SE", "free_SE", "free_SE", "free_SE", "free_SE"), to = c("free_NE", "free_NW", "free_E", "free_W", "free_SW", "curr_lane"))
bl_w <- data.frame(from = c("free_W", "free_W", "free_W", "free_W", "free_W", "free_W"), to = c("free_NE", "free_NW", "free_SW", "free_E", "free_SE", "curr_lane"))
bl_sw <- data.frame(from = c("free_SW", "free_SW", "free_SW", "free_SW", "free_SW", "free_SW"), to = c("free_NE", "free_NW", "free_W", "free_E", "free_SE", "curr_lane"))
bl_cl <- data.frame(from = c("curr_lane", "curr_lane", "curr_lane", "curr_lane", "curr_lane", "curr_lane"), to = c("free_NE", "free_NW", "free_W", "free_E", "free_SE", "free_SW"))
bl <- rbind(bl_nep, bl_nwp, bl_ep, bl_sep, bl_wp, bl_swp, bl_clp, bl_ne, bl_nw, bl_e, bl_se, bl_w, bl_sw, bl_cl)

condition1 <- str_detect(df$action, '^cruise$')
df1 <- subset(df, condition1)
df1 <- subset(df1, select=-c(action))
df1 <- df1[complete.cases(df1), ]
df1 <- correct_unique(df1)
df_factor <- as.data.frame(lapply(df1, factor))
network_structure <- hc(df_factor, whitelist = NULL, blacklist = bl, score = "bic", debug = FALSE, restart = 0, perturb = 1, max.iter = Inf, maxp = Inf, optimized = TRUE)
model_filename <- paste(output, "_cruise.dot", sep = "", collapse = "")
write.dot(network_structure, file = model_filename)
model_filename <- paste(output, "_cruise.png", sep = "", collapse = "")
png(model_filename)
plot(network_structure)
dev.off()
cmd <- paste("dot -Tps ", output, "_cruise.dot -o ", output, ".ps", sep = "", collapse = "")
system(cmd, intern = TRUE)
bn_fit <- bn.fit(network_structure, data = df_factor, method = "mle")
model_filename <- paste(output, "_cruise.net", sep = "", collapse = "")
write.net(model_filename, bn_fit)

file <- file(output, "w")
writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%", con = file)  
writeLines("% State variables", con = file)  
writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%\n", con = file)  
writeLines("state_fluent(curr_lane).", con = file)
writeLines("state_fluent(free_E).", con = file)
writeLines("state_fluent(free_NE).", con = file)
writeLines("state_fluent(free_NW).", con = file)
writeLines("state_fluent(free_SE).", con = file)
writeLines("state_fluent(free_SW).", con = file)
writeLines("state_fluent(free_W).", con = file)
writeLines("\n%%%%%%%%%%%%%%%%%%%%%%%%%%", con = file)
writeLines("% Actions", con = file)
writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%\n", con = file)
writeLines("action(cruise).", con = file)
writeLines("action(keep).", con = file)
writeLines("action(change_to_left).", con = file)
writeLines("action(change_to_right).", con = file)
writeLines("\n%%%%%%%%%%%%%%%%%%%%%%%%%%", con = file)  
writeLines("% Utilities", con = file)  
writeLines("%%%%%%%%%%%%%%%%%%%%%%%%%%\n", con = file)  
writeLines("utility(move_ahead_right, 20.0).", con = file)
writeLines("utility(move_ahead_left, 10.0).", con = file)
writeLines("utility(move_to_right, 30.0).", con = file)
writeLines("utility(rearEnd_crash, -200.0).", con = file)
writeLines("utility(sideSwipe_crash, -150.0).", con = file)
writeLines("utility(run_off_road, -200.0).\n", con = file)
writeLines("move_ahead_right :- curr_lane(0), free_NE(0), cruise.", con = file)
writeLines("move_ahead_left :- not(curr_lane(0)), free_NW(0), (not(free_NE(0)); not(free_E(0))), cruise.", con = file)
writeLines("move_to_right :- not(curr_lane(0)), free_NE(0), free_E(0), (free_NW(0); not(free_NW(0))), change_to_right.", con = file)
writeLines("rearEnd_crash :- not(curr_lane(0)), not(free_NW(0)), cruise.", con = file)
writeLines("rearEnd_crash :- curr_lane(0), not(free_NE(0)), cruise.", con = file)
writeLines("rearEnd_crash :- not(curr_lane(0)), not(free_NE(0)), change_to_right.", con = file)
writeLines("rearEnd_crash :- curr_lane(0), not(free_NW(0)), change_to_left.", con = file)
writeLines("sideSwipe_crash :- not(curr_lane(0)), not(free_E(0)), change_to_right.", con = file)
writeLines("sideSwipe_crash :- curr_lane(0), not(free_W(0)), change_to_left.", con = file)
writeLines("run_off_road :- not(curr_lane(0)), change_to_left.", con = file)
writeLines("run_off_road :- curr_lane(0), change_to_right.", con = file)
close(file)

action_str <- "cruise"
get_BN(bn_fit, action_str, output)

condition1 <- str_detect(df$action, '^change_to_left$')
df1 <- subset(df, condition1)
df1 <- subset(df1, select=-c(action))
df1 <- df1[complete.cases(df1), ]
df1 <- correct_unique(df1)
df_factor <- as.data.frame(lapply(df1, factor))
network_structure <- hc(df_factor, whitelist = NULL, blacklist = bl, score = "bic", debug = FALSE, restart = 0, perturb = 1, max.iter = Inf, maxp = Inf, optimized = TRUE)
model_filename <- paste(output, "_change_to_left.dot", sep = "", collapse = "")
write.dot(network_structure, file = model_filename)
model_filename <- paste(output, "_change_to_left.png", sep = "", collapse = "")
png(model_filename)
plot(network_structure)
dev.off()
cmd <- paste("dot -Tps ", output, "_change_to_left.dot -o ", output, ".ps", sep = "", collapse = "")
system(cmd, intern = TRUE)
bn_fit <- bn.fit(network_structure, data = df_factor, method = "mle")
model_filename <- paste(output, "_change_to_left.net", sep = "", collapse = "")
write.net(model_filename, bn_fit)
action_str <- "change_to_left"
get_BN(bn_fit, action_str, output)

condition1 <- str_detect(df$action, '^keep$')
df1 <- subset(df, condition1)
df1 <- subset(df1, select=-c(action))
df1 <- df1[complete.cases(df1), ]
df1 <- correct_unique(df1)
df_factor <- as.data.frame(lapply(df1, factor))
network_structure <- hc(df_factor, whitelist = NULL, blacklist = bl, score = "bic", debug = FALSE, restart = 0, perturb = 1, max.iter = Inf, maxp = Inf, optimized = TRUE)
model_filename <- paste(output, "_keep.dot", sep = "", collapse = "")
write.dot(network_structure, file = model_filename)
model_filename <- paste(output, "_keep.png", sep = "", collapse = "")
png(model_filename)
plot(network_structure)
dev.off()
cmd <- paste("dot -Tps ", output, "_keep.dot -o ", output, ".ps", sep = "", collapse = "")
system(cmd, intern = TRUE)
bn_fit <- bn.fit(network_structure, data = df_factor, method = "mle")
model_filename <- paste(output, "_keep.net", sep = "", collapse = "")
write.net(model_filename, bn_fit)
action_str <- "keep"
get_BN(bn_fit, action_str, output)

condition1 <- str_detect(df$action, '^change_to_right$')
df1 <- subset(df, condition1)
df1 <- subset(df1, select=-c(action))
df1 <- df1[complete.cases(df1), ]
df1 <- correct_unique(df1)
df_factor <- as.data.frame(lapply(df1, factor))
network_structure <- hc(df_factor, whitelist = NULL, blacklist = bl, score = "bic", debug = FALSE, restart = 0, perturb = 1, max.iter = Inf, maxp = Inf, optimized = TRUE)
model_filename <- paste(output, "_change_to_right.dot", sep = "", collapse = "")
write.dot(network_structure, file = model_filename)
model_filename <- paste(output, "_change_to_right.png", sep = "", collapse = "")
png(model_filename)
plot(network_structure)
dev.off()
cmd <- paste("dot -Tps ", output, "_change_to_right.dot -o ", output, ".ps", sep = "", collapse = "")
system(cmd, intern = TRUE)
bn_fit <- bn.fit(network_structure, data = df_factor, method = "mle")
model_filename <- paste(output, "_change_to_right.net", sep = "", collapse = "")
write.net(model_filename, bn_fit)
action_str <- "change_to_right"
get_BN(bn_fit, action_str, output)

options(error=NULL)
