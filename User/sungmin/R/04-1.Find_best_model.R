Output <- as.data.frame(matrix(,nrow = 0,ncol = 3))
column <- c("File","BestModel","Accuracy")
colnames(Output) <- c("File","BestModel","Accuracy")

for(i in 0:4){
  df <- read.csv(paste0("/home/tjahn/Data/prediction_find_best_model/prediction_",i,".csv"))
  object <- nrow(df)
  df <- df[,4:object]
  df <- as.data.frame(t(df))
  max_Accuracy <- max(df[,object])
  max_row_num <- which.max(df[,object])
  max_model <- rownames(df[max_row_num,object])
  add <- c(paste0("prediction_",i,".csv"),max_model,max_Accuracy)
  add <-t(matrix(add))
  colnames(add) <- column
  Output <- rbind(Output,add)
}
write.csv(Output,"/home/tjahn/Data/prediction_find_best_model/find_best_model.csv")
