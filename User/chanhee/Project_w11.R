setwd("/home/tjahn/Data")

multmerge = function(mypath){
  filenames=list.files(path=mypath, full.names=TRUE)
  datalist = lapply(filenames, function(x){read.table(file=x,header=T)})
  
  check<-lapply(datalist,dim)
  index<-sapply(check,function(x){x[1]==54675&&x[2]>3})
  datalist<-datalist[index]
  filelist<-filenames[index]
  
  cancer_or_not<-sapply(filelist,function(x){as.integer(substr(x,nchar(x)-4,nchar(x)-4))})
  num_of_patients<-sapply(datalist,function(x){dim(x)[2]-3})
  
  result<-mapply(function(x,y){rep(x,y)},cancer_or_not,num_of_patients)
  result<-unlist(result,use.names = FALSE)
  data<-Reduce(function(x,y){y<-y[c(-1,-2,-3)]; cbind(x,y)}, datalist)
  
  return(list(Data = data, Result = result))
  
}

# Merging files into one dataframe

datas<-multmerge("/home/tjahn/Data/cancer_normal_database/GEO_GPL570")
df<-datas$Data
result<-datas$Result

# Merging df and result, transposing data frame

df<-as.data.frame(t(df[,c(-1,-3)]))
colnames(df) <- as.character(unlist(df[1,]))
df = df[-1, ]
df$Result<-result

write.csv(df,"DRA_Data.csv")
