rm(list=ls())
library(pROC)
library(Metrics)
library(MLmetrics)
library(tidyverse)
library(mltools)
### Make the plot for the paper.
df <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(df) <- c('Method', 'MLAlg',  'Data', 'Run', 'F1_score')
type = "full"
dir_file = paste0("/data/hoan/attt/URLPhishing/comparision_results/", type, "prob")
# for (data_dir in list.files("/data/hoan/attt/URLPhishing/comparision_results", full.names =TRUE)){
for (data_dir in list.files(dir_file, full.names =TRUE)){
  # print(data_dir)
  info <- strsplit(data_dir, split = "/")
  data_info <- strsplit(dplyr::last(info[[1]]), split = "_")[[1]]
  output_labels <- read.table(data_dir, header = TRUE, sep = ",")
  true_labels <- as.integer(output_labels[,1])
  predict_labels <- as.integer(output_labels[,2])
  predict_scores <- as.numeric(output_labels[,3])
  if (length(true_labels)==length(predict_labels)){
    # if (grepl( "N1k", data_info[1], fixed = TRUE) & !(data_info[5] %in% c("GBDT", "NearestNeighbors"))){
      # grepl( needle, haystack, fixed = TRUE)
      print(data_dir)
      F1_value <- F1_Score(true_labels, predict_labels,1)
      # F1_value <- Metrics::f1(true_labels, predict_labels)
      recall_value <- Metrics::recall(true_labels, predict_labels)
      precision_value <- Metrics::precision(true_labels, predict_labels)
      # roc_value <- Metrics::auc(true_labels, predict_scores)
      roc_value <- auc_roc(true_labels, predict_scores)
      ## Change Matilda
      # print(data_dir)
      # if (data_info[5] != "test"){
      n_last = length(data_info)
      if (data_info[n_last-1]=="SimpleLSTM"){
        data_info[n_last-1] <- "LSTM"
      }
      if (data_info[n_last-1]=="LSTMConvFully"){
        data_info[n_last-1] <- "LSTMMultiConv"
      }
      if (data_info[n_last-1]=="PredictionsSimpleLSTM3"){
        data_info[n_last-1] <- "LSTMCNNv2"
      }
      if (data_info[n_last-1]=="NearestNeighbors"){
        data_info[n_last-1] <- "K-NN"
      }
      df <- rbind(df, list(Method=data_info[n_last-1], Data=data_info[1], Run=data_info[6], F1_score=F1_value, Precision=precision_value, Recall=recall_value, AUC = roc_value))
      # }
    # }
  }
}

ggplot(df, aes(x=Method, y=F1_score, fill=Method)) +
  geom_boxplot()
ggplot(df, aes(x=Method, y=Precision, fill=Method)) +
  geom_boxplot()
ggplot(df, aes(x=Method, y=Recall, fill=Method)) +
  geom_boxplot()
ggplot(df, aes(x=Method, y=AUC, fill=Method)) +
  geom_boxplot()

p0 <- ggplot(data=df, aes(x=Method, y=F1_score, color=Method)) +
  geom_boxplot(outlier.shape = NA, show.legend = TRUE) + theme_minimal() + theme_classic() +
  labs(x = "", y = "F1-score") +   theme(axis.text=element_text(colour="black"),  axis.text.x = element_text(angle = 45, vjust = 0.8, hjust=1))# + ylim(0, 1)
ggsave(filename = paste0("/data/hoan/attt/URLPhishing/plots/F1score",type,".pdf"), p0, width = 5, height = 3, dpi = 300, units = "in", device='pdf')

p1 <- ggplot(data=df, aes(x=Method, y=Precision, color=Method)) +
  geom_boxplot(outlier.shape = NA, show.legend = TRUE) + theme_minimal() + theme_classic() +
  labs(x = "", y = "Precision") +   theme(axis.text=element_text(colour="black"),  axis.text.x = element_text(angle = 45, vjust = 0.8, hjust=1))# + ylim(0, 1)
ggsave(filename = paste0("/data/hoan/attt/URLPhishing/plots/Precision",type,".pdf"), p1, width = 5, height = 3, dpi = 300, units = "in", device='pdf')

p2 <- ggplot(data=df, aes(x=Method, y=Recall, color=Method)) +
  geom_boxplot(outlier.shape = NA, show.legend = TRUE) + theme_minimal() + theme_classic() +
  labs(x = "", y = "Recall") +   theme(axis.text=element_text(colour="black"),  axis.text.x = element_text(angle = 45, vjust = 0.8, hjust=1))# + ylim(0, 1)
ggsave(filename = paste0("/data/hoan/attt/URLPhishing/plots/Recall",type,".pdf"), p2, width = 5, height = 3, dpi = 300, units = "in", device='pdf')

p3 <- ggplot(data=df, aes(x=Method, y=AUC, color=Method)) +
  geom_boxplot(outlier.shape = NA, show.legend = TRUE) + theme_minimal() + theme_classic() +
  labs(x = "", y = "AUC") +   theme(axis.text=element_text(colour="black"),  axis.text.x = element_text(angle = 45, vjust = 0.8, hjust=1))# + ylim(0, 1)
ggsave(filename = paste0("/data/hoan/attt/URLPhishing/plots/AUC",type,".pdf"), p3, width = 5, height = 3, dpi = 300, units = "in", device='pdf')

dfavg <- aggregate(df$F1_score, list(df$Method), FUN=mean, na.rm = TRUE)
print(dfavg)
dfavg <- aggregate(df$Precision, list(df$Method), FUN=mean, na.rm = TRUE)
print(dfavg)
dfavg <- aggregate(df$Recall, list(df$Method), FUN=mean, na.rm = TRUE)
print(dfavg)
dfavg <- aggregate(df$AUC, list(df$Method), FUN=mean, na.rm = TRUE)
print(dfavg)
