#setwd("/home/tjahn/Data/")
forms <- c("FPKM-UQ","FPKM","htseq")
#input <- "/home/tjahn/Data/01.4964/"
#output<- "/home/tjahn/Data/sungminTCGAtoGEO_2500/"
for(form in forms){
  
  df<-read.csv(paste0("/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final/Final_TCGA_gene_expression_",form,".csv"), header=TRUE)
  names(df) <- gsub(".", "-", names(df), fixed = TRUE)
  print(paste0("large table : ", form, " reads."))
  
  #TCGA_2500 <-read.csv(paste0("/home/tjahn/Data/sungminTCGAtoGEO_2500/TCGA_",form,"_2500_names.csv"))
  #TCGA_2500 <-as.character(TCGA_2500$x)
  #TCGA_2500 <-gsub(".", "-", TCGA_2500, fixed = TRUE)
  #GEO_2500 <- read.csv("GEO_2500_names.csv", header = TRUE)
  #GEO_2500 <- as.character(GEO_2500$x)
  #GEO_2500 <-  gsub(".", "-", GEO_2500, fixed = TRUE)
  gene_2500 <- read.csv("/home/tjahn/Data/sungminTCGAtoGEO_2500/genenames_2500.txt", header = F)
  gene_2500 <- as.character(gene_2500$x)
  gene_2500 <-  gsub(".", "-", gene_2500, fixed = TRUE)
  
  genes <- df[,2:(ncol(df)-3)]
  patients <- df$file_name
  result <- df$result
  cancer_code <- df$cancer_code
  index <- df$index
  genes <- subset(genes, select = gene_2500)
  df_ch <- cbind(patients, genes, result, cancer_code, index)
  print("df modified.")
  
  write.csv(df_ch, paste0("/home/tjahn/Data/genename_2500_by_Goo/TCGA_",form,"_2500_by_Goo.csv"), row.names=FALSE)
  print(paste0("modified table: ", form, " written."))
}
