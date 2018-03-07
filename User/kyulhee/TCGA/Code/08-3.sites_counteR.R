#setwd("C:\\test\\sam")
setwd("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final")

forms <- c("FPKM-UQ", "FPKM", "htseq")

for(form in forms){
  
  print(paste0(form, " @@ type counting starts."))
  
  csv_file <- paste0("wt_type_Final_TCGA_gene_expression_", form, ".csv")
  #csv_file <- "wt_type_Final_TCGA_gene_expression_FPKM-UQ_4964_ch.csv"
  large_table <- read.csv(csv_file, header = TRUE)
  print("large table reads.")
  
  type_temp <- data.frame(table(large_table$types))
  names(type_temp)[1] <- "types"
  names(type_temp)[length(type_temp[1,])] <- form
  print(paste0(form, " @@ type counting complete."))
  
  write.csv(type_temp, paste0("/home/tjahn/Git2/User/kyulhee/TCGA/", form, "_types_counting.csv"), row.names = FALSE)
  #write.csv(type_temp, "types_counting.csv", row.names = FALSE)
  print("counting file written.")      
  
}

