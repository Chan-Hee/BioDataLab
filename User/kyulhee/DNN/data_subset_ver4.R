library(csv)

######### Read csv file with header #########
#file_ori <- read.csv("/home/tjahn/Data/FinalData_GSM_gene_index_result.csv",header=TRUE) 
#Running in server, fail
file_ori <- read.csv("FinalData_GSM_gene_index_result.csv",header=TRUE)

######### make list for 'gene_off' ########
#conf_directory = "/home/tjahn/Git2/User/kyulhee/DNN/input/"
#conf_filename = "input.csv"
#conf = read.csv(paste()(conf_directory, conf_filename,collapse=NULL)
#Running in server, fail
conf <- c(0,10,20,30,40,50,60,70,80,90,95,99)

######## for reproduction ########                
set.seed(777)
                
######## extract header & set gene length ########
file_header <- names(file_ori)
gene_len <- length(file_header)-3

######## make csv file ########
for(i in conf){
  #file header off : concatenate three header(s)- patient name(file_header[1]), sampled genes(fileheader[sample~]), gene index & result(file_header[6002:6003]) 
  file_header_off <- c(file_header[1], file_header[sample(gene_len,round(gene_len*(1-0.01*i),digits=0))+1], file_header[6002:6003])
  file_name <- paste0("FinalData_Random_", i, "off_GSM_gene_index_result.csv", collapse = NULL)
  #write csv : extract sampled columns from original file using file_header_off
  write.csv(subset(file_ori, select=file_header_off), row.names=FALSE, file_name)
}