

# Data Merging week 11 homework #
#         What to do !!!!       #

## Merge All files into data frame
## Save data about whehter the paitent is cancer or not



#Environment Setting



# Merging files into one dataframe
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

datas<-multmerge("/home/tjahn/Data/cancer_normal_database/GEO_GPL570")
#datas<-multmerge("/Users/chanhee/Desktop/Data")

df<-datas$Data
result<-datas$Result






# Data Preprocessing week 12 homework #
#         What to do !!!!             #

## Genes with no names, Genes that are duplicated
## Remove row1, row3, and then transpose the data
## Add result to the dataframe
## Explore the Data!
## Remove Na Data
## Set the data type correctly for result and gene data
## Normalize the data 





p_df<-df[df$Gene_Symbol!="",] 
p_df<-p_df[!duplicated(p_df[,2]),]

p_df<-as.data.frame(t(p_df[,c(-1,-3)]))
colnames(p_df) <- as.character(unlist(p_df[1,]))
p_df = p_df[-1, ]

p_df$Result<-result

not_na<-apply(p_df,1,function(x){!any(is.na(x))})
p_df<-p_df[not_na,]

p_df$Result<-as.factor(p_df$Result)

num_val<-apply(p_df[,-length(p_df)],2,function(x){round(as.numeric(as.character(x)),digits = 3)})
p_df[,-length(p_df)]<-as.data.frame(num_val)

#distribution0<-apply(p_df[p_df$Result=="0",-length(p_df)],1,mean)
#distribution1<-apply(p_df[p_df$Result=="1",-length(p_df)],1,mean)
#distribution
#plot(distribution0)
#points(distribution1,col = "red")

p_df[,-length(p_df)]<-apply(p_df[,-length(p_df)],1,function(x){round((x-mean(x))/sd(x),digits = 3)})

#distribution0<-apply(p_df[p_df$Result=="0",-length(p_df)],1,mean)
#distribution1<-apply(p_df[p_df$Result=="1",-length(p_df)],1,mean)
#distribution
#plot(distribution0)
#points(distribution1,col = "red")
p_df$Index<-sample(1:5,dim(p_df)[1],replace = TRUE)
write.csv(p_df,"FinalData2.csv")
