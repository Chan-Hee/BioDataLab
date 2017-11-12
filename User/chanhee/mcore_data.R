library(plyr)
library(parallel)
rm(list=ls())
gc()
start<-proc.time()
multmerge = function(mypath){
  filenames=list.files(path=mypath, full.names=TRUE)
  datalist = mclapply(filenames, function(x){read.table(file=x,header=T)},mc.cores=20)
  
  check<-mclapply(datalist,dim)
  index<-unlist(mclapply(check,function(x){x[1]==54675&&x[2]>3},mc.cores=20))
  
  datalist<-datalist[index]
  filelist<-filenames[index]
  
  cancer_or_not<-unlist(mclapply(filelist,function(x){as.integer(substr(x,nchar(x)-4,nchar(x)-4))},mc.cores=20))
  num_of_patients<-unlist(mclapply(datalist,function(x){dim(x)[2]-3},mc.cores=20))
  
  result<-mcmapply(function(x,y){rep(x,y)},cancer_or_not,num_of_patients,mc.cores=20)
  result<-unlist(result,use.names = FALSE)
  data<-Reduce(function(x,y){y<-y[c(-1,-2,-3)]; cbind(x,y)}, datalist)
  
  return(list(Data = data, Result = result))
}

datas<-multmerge("/home/tjahn/Data/cancer_normal_database/GEO_GPL570")
print("multmerge finished")

df<-datas$Data
result<-datas$Result

p_df<-df[df$Gene_Symbol!="",] 
p_df<-p_df[!duplicated(p_df[,2]),]

print("transpose begin")

p_df<-as.data.frame(t(p_df[,c(-1,-3)]))
colnames(p_df) <- as.character(unlist(p_df[1,]))
p_df = p_df[-1, ]

p_df$Result<-result

print("Remove NA")

not_na<-apply(p_df,1,function(x){!any(is.na(x))})
p_df<-p_df[not_na,]

p_df$Result<-as.factor(p_df$Result)

print("Normalize")

num_val<-apply(p_df[,-length(p_df)],2,function(x){round(as.numeric(as.character(x)),digits = 3)})
p_df[,-length(p_df)]<-as.data.frame(num_val)


p_df[,-length(p_df)]<-apply(p_df[,-length(p_df)],1,function(x){round((x-mean(x))/sd(x),digits = 3)})

end<-proc.time()
time<-start-end
print(time)

write.csv(p_df,"Final_MCore_Data.csv")  
