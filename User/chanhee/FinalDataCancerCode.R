
# Merging files into one dataframe
multmerge = function(mypath){
  filenames=list.files(path=mypath, full.names=TRUE)
  datalist = lapply(filenames, function(x){read.table(file=x,header = T)})
  
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

datas<-multmerge("/home/data/GEO_GPL570")
#datas<-multmerge("/Users/chanhee/Desktop/Data")

df<-datas$Data

print("Data Before Preprocessed: nrow = genes, ncol = GeneID(1,2,3)+patients")
print(dim(df))

## Save data about whehter the paitent is cancer or not
result<-datas$Result

# Genes with no names, Genes that are duplicated
p_df<-df[df$Gene_Symbol!="",] 
p_df<-p_df[!duplicated(p_df[,2]),]

# Remove row1, row3, and then transpose the data
p_df<-as.data.frame(t(p_df[,c(-1,-3)]))
colnames(p_df) <- as.character(unlist(p_df[1,]))
p_df = p_df[-1, ]

# Add result to the dataframe
p_df$Result<-result

## Remove Na Data
not_na<-apply(p_df,1,function(x){!any(is.na(x))})
p_df<-p_df[not_na,]


# Set the data type correctly for result and gene data
p_df$Result<-as.factor(p_df$Result)

num_val<-apply(p_df[,-length(p_df)],2,function(x){round(as.numeric(as.character(x)),digits = 3)})
p_df[,-length(p_df)]<-as.data.frame(num_val)

#remove no name patients
p_df<-p_df[substr(row.names(p_df),1,3)=="GSM",]



# change the non log 2 scaled data to log2 scale



maxs<-apply(p_df[,-length(p_df)],1,max)
not_log2_scale_ids = names( which(maxs > 100 ) )

for(i in 1:length(not_log2_scale_ids)){
  exception = p_df[not_log2_scale_ids[i],-length(p_df)]<1
  p_df[not_log2_scale_ids[i],exception] = 1
  temp = log2( p_df[not_log2_scale_ids[i],-length(p_df) ] )
  p_df[not_log2_scale_ids[i],-length(p_df)] = temp
}


# Normalize the data 

for(i in 1:dim(p_df)[1]){
  p_df[i,-length(p_df)] <- (p_df[i,-length(p_df)]-mean(unlist(p_df[i,-length(p_df)])))/sd(p_df[i,-length(p_df)])
}

print("Data after preprocessed: nrow = patients, ncol = genes + Result")
print(dim(p_df))


# Get Variance of each gene and feature select!
GeneVar<-apply(p_df[,-length(p_df)],2,sd)
GeneVar<-GeneVar[rev(order(GeneVar))]
GeneVar<-GeneVar[1:6000]
temp<-p_df[,names(GeneVar)]
temp$Result<-p_df$Result
p_df<-temp

#Add Cancer Case
CancerCode<-read.csv("/home/tjahn/Git/User/chanhee/GPL570_sampleinfo.txt",sep = "\t")
row.names(CancerCode) = CancerCode$GSM_ID
GSM<-row.names(p_df)
CancerCode<-CancerCode[GSM,]
p_df$CancerCode<-CancerCode$CANCER_CODE

# Add Five-fold  index
p_df$Index<-sample(1:5,dim(p_df)[1],replace = TRUE)

print("Final Data: nrow = patients, ncol = genes(6000) + Result + CancerCode + index")
print(dim(p_df))

write.csv(p_df,"Data_with_CancerCode.csv")

