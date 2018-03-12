#################################### read & merge datas ####################################
handleData <- function(ref, wd, out){

  ref_table <- read.csv(ref, header=TRUE)
  ref_table <- ref_table[unique(ref_table$Ensembl.ID.supplied.by.Ensembl.),]
  
  TCGA_genes <- as.character(ref_table$Approved.Symbol)
  gene_ids <- as.character(ref_table$Ensembl.ID.supplied.by.Ensembl.)
  
  TCGA_table_frame <- data.frame(TCGA_genes)
  TCGA_table <- TCGA_table_frame
  
  setwd(wd)
  file_list = c(list.files(pattern="*.counts"),list.files(pattern="*.txt"))
  file_form <- NULL
  for(f in file_list){
    file_form <- c(file_form,unlist(strsplit(f, ".", fixed=TRUE))[2])
  }
  
  file_table <- data.frame(file_list, file_form)
  #forms <- c("FPKM-UQ", "FPKM", "htseq")
  forms <- "htseq"
  print("***************************  file table done.")
  for(form in forms){
    print(paste0("***************************  format: @ ", form, " @ starts."))
    file_list <- file_table[file_table$file_form==form,1]
    file_num <- 0
    error_files <- NULL
    
    if(form == "htseq")
    {
      ref_index <- (ref_table$index)+5
    }else
    {
      ref_index <- ref_table$index
    }
    # htseq have 5 more elements. Add 5 to index for move back reading frame.
    
    #for(i in 1:length(file_list)){
    i=1
      
      exp_file <- file(as.character(file_list[i]))
      table_temp <- read.table(exp_file, header = FALSE)
      table_temp <- table_temp[order(table_temp$V1), ]
      write.csv(table_temp,paste0(out,"\\TCGA_",i,"_gene_ids.csv"), row.names = FALSE)
      table_temp <- table_temp[ref_index, ]
      write.csv(table_temp,paste0(out,"\\TCGA_",i,"_gene_ids_parsing.csv"), row.names = FALSE)
      id_sets <-
        strsplit(as.character(table_temp[, 1]), ".", fixed = TRUE)
      ids <- NULL
      for (k in 1:length(id_sets)) {
        ids <- c(ids, unlist(id_sets[k])[1])
      }
      table_temp <- data.frame(ids, table_temp$V2)
      names(table_temp) <- c("ids", as.character(file_list[i]))
      
      if(isErrorfile(table_temp, gene_ids)){
        error_files <- c(error_files, as.character(file_list[i]))
      }else{
        TCGA_table <- cbind(TCGA_table,table_temp[2])
        names(TCGA_table)[length(TCGA_table[1,])] <- as.character(file_list[i])
      }
      
      print(paste0(i,"/",length(file_list)," file done."))
    
      
      if(i%%100==0 || i==length(file_list)){
        
        file_num <- file_num+1
        file_name <- names(TCGA_table)[-1] 
        #log2 scale
        maxs <- apply(TCGA_table[,-1],2,max)
        not_log2_scale_ids <- names( which(maxs > 100 ) )
        for(j in 1:length(not_log2_scale_ids)){
          exception = TCGA_table[,not_log2_scale_ids[j]]<1
          TCGA_table[exception,not_log2_scale_ids[j]] = 1
          temp = log2(TCGA_table[,not_log2_scale_ids[j]])
          TCGA_table[,not_log2_scale_ids[j]] = temp
        }
        print(paste0("***************************  ",as.character(file_num)," file log2 done."))
        
        # Normalize
        for(k in 2:dim(TCGA_table)[2]){
          TCGA_table[,k] <- (TCGA_table[,k]-mean(TCGA_table[,k]))/sd(TCGA_table[,k])
        }
        print(paste0("***************************  ",as.character(file_num)," file normalized."))
        
        # Transpose
        TCGA_table <- data.frame(t(TCGA_table))
        TCGA_table <- TCGA_table[-1,]
        names(TCGA_table) <- TCGA_genes
        TCGA_table <- cbind(file_name, TCGA_table)
        print(paste0("***************************  ",as.character(file_num)," file transposed."))
        
        # Add TCGA barcode
        #barcode <- getBarcode(TCGA_table$file_name, "C:\\test\\merge_pro.csv")
        barcode <- getBarcode(TCGA_table$file_name, "/home/tjahn/Git2/User/kyulhee/TCGA/merge_pro.csv")
        TCGA_table <- cbind(TCGA_table, barcode)
        print(paste0("***************************  ",as.character(file_num)," file added barcode."))
        
        # Add Tumor result
        result <- tumorDisc(TCGA_table$sample_id)
        TCGA_table <- cbind(TCGA_table, result)
        print(paste0("***************************  ",as.character(file_num)," file added result."))
        
        # Write file(has 100 samples)
        #csv_file_name = paste0(out, "\\TCGA_genes_",form,"_",as.character(file_num),".csv")
        csv_file_name = paste0(out, "/TCGA_genes_",form,"_",as.character(file_num),".csv")
        write.csv(TCGA_table, csv_file_name, row.names = FALSE)
        print(paste0("***************************  ",as.character(file_num)," file written."))
        
        #table initialization
        TCGA_table <- TCGA_table_frame
        
        
      }
    #}
    if(length(error_files)>0){
      print(paste0(form, " error file detect: ", length(error_files)))
            #error_csv_name <- paste0(out, "\\TCGA_genes_",form,"_error",".csv")
            error_csv_name <- paste0(out, "/TCGA_genes_",form,"_error",".csv")
            write.csv(error_files, error_csv_name, row.names = FALSE)
    }else{
      print(paste0(form, " has no error file."))
    }
    
  }
}


#################################### check errors ####################################

isErrorfile <- function(table_temp, gene_ids){
  
  error_switch <- 0
  
  for (t in 1:length(table_temp$ids)) {
    if (table_temp$ids[t] != gene_ids[t]){
      print(paste0(
        "@@@@@@@@@@@@@@@@@@@@@@@ Error detect! file: ",
        names(table_temp)[2]
      ))
      error_switch <- error_switch+1
    }
  }
  if(error_switch>0)
  {
    return(1)
  }else
  {
    return(0)
  }
}


#################################### getting TCGA barcode ####################################

getBarcode <- function(file_names, barcode_file){
  
  
  barcode_ref <- read.csv(barcode_file)
  barcode <- NULL
  for(i in file_names){
    barcode <- rbind(barcode, subset(barcode_ref, barcode_ref$file_name %in% i)[2:3] )   
  }
  return(barcode)
  
}



#################################### discrimination Normal/Tumor ####################################

tumorDisc <- function(sample_id){
  
  result <- as.integer(!(sample_id %in% 10:19))
  # 01~09 -> tumor, 10~19 -> normal, 40 -> special tumor case
  return(result)
  
} 



#################################### Main ####################################
#ref = "/home/tjahn/Git2/User/kyulhee/TCGA/hgnc_symbols_ref_inter.csv"
ref = "C://test//hgnc_symbols_ref_inter.csv"
# where's "hgnc_symbols_ref_inter.csv" file?
#wd = "/home/tjahn/GDC_Data/GeneExpression/Gene_txt_files"
wd = "C://test//exp//error"
# where're expression files? 
#out = "/home/tjahn/GDC_Data/GeneExpression/Gene_txt_sum"
out = "C://test//sam"
# output file space
handleData(ref,wd,out)
