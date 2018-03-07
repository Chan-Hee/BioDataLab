ref = "/home/tjahn/Git2/User/kyulhee/TCGA/hgnc_symbols_ref_inter.csv"
#ref = "C://test//hgnc_symbols_ref_inter.csv"
# where's "hgnc_symbols_ref_inter.csv" file?
wd = "/home/tjahn/GDC_Data/GeneExpression/Gene_txt_files"
#wd = "C://test//exp"
# where're expression files?
out = "/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum/Final"
#out = "C://test//sam"
# output file space

ref_table <- read.csv(ref, header = TRUE)
#ref_table <- read.csv("C:\\test\\hgnc_symbols_ref_inter.csv", header=TRUE)
ref_table <-
  ref_table[unique(ref_table$Ensembl.ID.supplied.by.Ensembl.), ]

TCGA_genes <- as.character(ref_table$Approved.Symbol)
gene_ids <- as.character(ref_table$Ensembl.ID.supplied.by.Ensembl.)

setwd(wd)
file_list = c(list.files(pattern = "*.counts"), list.files(pattern = "*.txt"))
file_form <- NULL
for (f in file_list) {
  file_form <- c(file_form, unlist(strsplit(f, ".", fixed = TRUE))[2])
}

file_table <- data.frame(file_list, file_form)
forms <- c("FPKM-UQ", "FPKM", "htseq")
print("***************************  file table done.")

for (form in forms) {
  print(paste0("***************************  format: @ ", form, " @ starts."))
  file_list <- file_table[file_table$file_form == form, 1]
  error_switch <- NULL
  
  if(form == "htseq"){
    ref_index <- ref_table$index+5
  }
  else{
    ref_index <- ref_table$index
  }
  # htseq have 5 more elements. Add 5 to index for move back reading frame.  
  
  for (i in 1:length(file_list)) {
    exp_file <- file(as.character(file_list[i]))
    table_temp <- read.table(exp_file, header = FALSE)
    table_temp <- table_temp[order(table_temp$V1), ]
    id_sets <-
      strsplit(as.character(table_temp[ref_index, 1]), ".", fixed = TRUE)
    ids <- NULL
    for (k in 1:length(id_sets)) {
      ids <- c(ids, unlist(id_sets[k])[1])
    }
    
    for (t in length(ids)) {
      if (ids[t] == gene_ids[t]) {
        error_switch <- c(error_switch, 0)
      }
      else{
        print(paste0(
          "@@@@@@@@@@@@@@@@@@@@@@@ Error detect! file: ",
          names(t)
        ))
        error_switch <- c(error_switch, 1)
      }
    }
  }
  
  error_table <- data.frame(file_list, error_switch)
  # Write file(has 100 samples)
  csv_file_name <- paste0("error_table_", form, ".csv")
  
  write.csv(error_table, csv_file_name, row.names = FALSE)
}
