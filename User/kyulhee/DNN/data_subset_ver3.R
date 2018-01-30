library(csv)

file_ori <- read.csv("/home/tjahn/Data/FinalData_GSM_gene_index_result.csv",header=TRUE)

conf_directory = "/home/tjahn/Git2/User/kyulhee/DNN/input/"
conf_filename = "input.csv"
conf = read.csv(paste()(conf_directory, conf_filename,collapse=NULL)
                
set.seed(777)
                
file_header <- names(file_ori)
gene_len <- length(file_header)-3

for(i in conf){
  file_header_off <- c(file_header[1], file_header[sample(gene_len,round(gene_len*(1-0.01*i),digits=0))+1], file_header[6002:6003])
  file_name <- paste0("FinalData_Random_", i, "off_GSM_gene_index_result.csv", collapse = NULL)
  write.csv(subset(file_ori, select=file_header_off), row.names=FALSE, paste()("/home/tjahn/Data/",file_name, collapse=NULL))
}