#setwd("C:\\test\\sam")
setwd("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final")

#ref <- read.csv("C:\\test\\TCGA_GEO_pro.csv", header = TRUE, na.strings = "NNN")
ref <- read.csv("/home/tjahn/Git2/User/kyulhee/TCGA/TCGA_GEO_pro.csv", header = TRUE, na.strings = "NNN")

forms <- c("FPKM-UQ", "FPKM", "htseq")
csv_files <- paste0("Final_TCGA_gene_expression_", forms, "_4964.csv")
#csv_files <- paste0("Final_TCGA_gene_expression_", forms, ".csv")
#csv_files <- paste0("Final_TCGA_gene_expression_", forms, "_4964_ch_reduced_final.csv")


for(i in csv_files){
  df <- read.csv(i, header = TRUE)
  print("csv file reads.")
  
  cancer_code <- NULL
  result <- df$result
  patient <- df$file_name
  for(j in 1:nrow(df)){
    cancer_code <- c(cancer_code, as.character(ref[ref[,1] == unlist(strsplit(as.character(df[j,df$barcode]), "-"))[2],3]))
  }
  print("code finding done.")
  
  with_code <- cbind(patient, csv_file[,2:(ncol(csv_file)-3)], result, cancer_code)
  names(with_code) <- gsub(".", "-", names(with_code), fixed = TRUE)
  
  #shuffle & add index
  with_code <- with_code[sample(nrow(with_code)),]
  remain <- nrow(with_code)%%5
  index <- rep(1:5,nrow(with_code)%/%5)
  if(remain>0){ index <- c(index, 1:remain) }
  with_code <- cbind(with_code, index)

    print("csv file modified.")
  
  file_name <- paste0("wt_cancer_code_",i)
  write.csv(with_code, file_name, row.names = FALSE)
  print("********************csv file written.")
}
  
