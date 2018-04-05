for (i in 0:4){
  #df = read.csv(paste0("D:/biodatalab/2018-1/prediction_output_each_GEO_in_each_model/prediction_output",i,"_data.csv"),sep = ",",header = T)
  df <- read.csv(paste0("/home/tjahn/FINALMODEL/BioDataLab_ML/prediction_output/prediction_output",i,"_data.csv"),sep = ",",header = T)
  #df_prediction = round(subset(df,select = -c(1:3)),0)
  df <- cbind(df[,2:3],round(subset(df,select = -c(1:3)),0))
  object <- 4408
  for(j in 3:ncol(df)){
    correct <- 0
    for(k in 1:object){
      if(df[k,j] == df[k,"result"]){
        correct <- correct + 1
      }else{
        correct <- correct + 0
      }
    }
    df[(object+1),j] <- correct/object
  }
  
  write.csv(df,paste0("/home/tjahn/Data/prediction_find_best_model/prediction_",i,".csv"))
  #write.csv(paste0("D:/biodatalab/2018-1/prediction_output_each_GEO_in_each_model/output/prediction_output",i,"_data.csv"))
}

#df = read.csv(paste0("D:/biodatalab/2018-1/prediction_output_each_GEO_in_each_model/prediction_output",i,"_data.csv"),sep = ",",header = T)
#df <- cbind(df[,2:3],round(subset(df,select = -c(1:3)),0))
#correct = 0
#for(j in 3:ncol(df)){
#  for(k in 1:nrow(df)){
#    if(df[k,j] == df[k,"result"]){
#      correct = correct + 1
#    }else{
#      correct = correct + 0
#    }
#  }
#  df[(nrow(df)+1),j] <- nrow(df)/correct
#}
