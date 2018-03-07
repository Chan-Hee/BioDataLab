setwd("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final")
#setwd("C:\\test\\sam")
ref <- read.csv("/home/tjahn/Git2/User/kyulhee/TCGA/inter_TCGA_GEO_source.csv", header=FALSE)
#ref <- read.csv("C:\\test\\inter_TCGA_GEO_source.csv", header=FALSE)
file_list <- list.files(pattern = "*.csv")
#i<- "wt_cancer_code_Final_TCGA_gene_expression_FPKM_4964_sam.csv"
for(i in file_list){
  df <- read.csv(i, header = TRUE)
  print("table reads.")
  names(df) <- gsub(".", "-", names(df), fixed = TRUE)
  #subsample
  df_ch <- df[as.character(df$cancer_code) %in% ref$V2,]
  #shuffle
  df_sh <- df_ch[sample(nrow(df_ch)),]
  
  #add index
  remain <- nrow(df_sh)%%5
  index <- rep(1:5,nrow(df_sh)%/%5)
  if(remain>0){ index <- c(index, 1:remain) }
  df_sh$index <- index 
  print("table subsampled.")
  
  file_name <- paste0("inter_", i)
  write.csv(df_sh, file_name, row.names = FALSE)
  print("***************** file written.")
}
