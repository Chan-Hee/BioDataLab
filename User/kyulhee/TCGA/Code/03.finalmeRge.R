#setwd("C:\\test\\sam")
setwd("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum")
file_list = list.files(pattern="*.csv", full.names = TRUE)
file_form <- NULL
for(f in file_list){
  file_form <- c(file_form,unlist(strsplit(f, "_"))[3])
}
file_table <- data.frame(file_list, file_form)
forms <- c("FPKM-UQ", "FPKM", "htseq")

for(form in forms){
  file_list <- as.character(file_table[file_table$file_form==form,1])
  print(paste0("***************************  format: @ ", form, " @ starts."))
  
  for(i in 1:length(file_list)){
    temp <- read.csv(file_list[i],header = TRUE)
    if(i==1){
      dat <- temp
    }
    else{
      dat <- rbind(dat, temp)
    }
    print(paste0(i,"/",length(file_list)," file done."))
  }
  print(paste0("***************************  format: @ ", form, " @ file reads."))
  #file_name <- paste0("C:\\test\\TCGA_gene_expression_", form, ".csv")
  file_name <- paste0("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final/Final_TCGA_gene_expression_", form, ".csv")
  write.csv(dat,file_name,row.names=FALSE)
}  
