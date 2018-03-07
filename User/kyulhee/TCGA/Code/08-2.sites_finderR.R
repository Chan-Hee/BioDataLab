#setwd("C:\\test\\sam")
setwd("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final")
sites <- read.csv("/home/tjahn/Git2/User/kyulhee/TCGA/Sites.csv", header = TRUE)
#sites <- read.csv("C:\\test\\Sites.csv", header = TRUE)

forms <- c("FPKM-UQ", "FPKM", "htseq")
#csv_files <- paste0("Final_TCGA_gene_expression_", forms, "_4964.csv")
csv_files <- paste0("Final_TCGA_gene_expression_", forms, ".csv")
#csv_files <- paste0("Final_TCGA_gene_expression_", forms, "_4964_ch_reduced_final.csv")
print("csv file reads.")

for(i in csv_files){
  csv_file <- read.csv(i, header = TRUE)
  #csv_file$file_name <- as.character(csv_file$file_name)
  sites$samples <- as.character(sites$samples)
  print("type finding starts.")
  types <- NULL
  for(j in 1:length(csv_file$file_name)){
    types <- c(types, as.character(sites[sites[,1] == csv_file[j,1],2]))
  }
  print("type finding done.")
  with_type <- cbind(csv_file, types)
  print("csv file modified.")
  file_name <- paste0("wt_type_",i)
  write.csv(with_type, file_name, row.names = FALSE)
  print("********************csv file written.")
}

