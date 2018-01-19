################################################### Merging Files #########################################

MergeUntil<-function(filenames){
  
  data<-read.table(filenames[1],header=TRUE)
  GSM<-colnames(data[4:length(data)])
  diagnose<-rep(as.integer(substr(filenames[1],nchar(filenames[1])-4,nchar(filenames[1])-4),times=length(GSM)))
  cancer<-data.frame(GSM_NUMBER=GSM,CANCER=diagnose)
  
  i=2
  while(i<length(filenames)){
    
    Ndata<-read.table(filenames[i],header=TRUE)
    
    if(nrow(Ndata)==nrow(data) && all(data$Gene_Symbol==Ndata$Gene_Symbol)){
      data<-cbind(data,Ndata[,c(-1,-2,-3)])
      NGSM<-colnames(Ndata[4:length(Ndata)])
      Ndiagnose<-rep(as.integer(substr(filenames[i],nchar(filenames[i])-4,nchar(filenames[i])-4),times=length(NGSM)))
      Ncancer<-data.frame(GSM_NUMBER=NGSM,CANCER=Ndiagnose)
      cancer<-rbind(cancer,Ncancer)
    }
    
    i=i+1
    #drop sample which contains at least one NA
    
    index = colSums(is.na(data)) == 0
    end = length(data)
    Rindex = index[4:end]
    
    data = data[ , index]
    
    #drop samples in CancerResult also
    
    cancer=cancer[Rindex,]
  }
  return(list(x=data,y=cancer))
}

MergeToyFile<-function(mypath){
  
  filenames<-list.files(path = mypath,full.names = TRUE)
  datalist <- MergeUntil(filenames)
  
  return(datalist)
}
################################################### Normalization #########################################

NormalizeToy<-function(RawToy){
  for(i in 4:length(colnames(RawToy))){
    RawToy[,i]<-(RawToy[,i]-mean(RawToy[,i]))/sd(RawToy[,i])
  }
  
  return(RawToy)
}
#################################################### VarianceTest ##########################################
GetVar<-function(Toy1000){
  Toy1000$VAR<-apply(Toy1000[,4:length(Toy1000)],1,sd)
  return(Toy1000)
}

#################################################### Main ##################################################
set.seed(777)
#for reproduction
setwd("/home/tjahn/Data/")
#datas<-MergeToyFile("/Users/chanhee/Desktop/Data")
datas<-MergeToyFile("/home/tjahn/Data/cancer_normal_database/GEO_GPL570")

RawToy<-datas$x
CancerResult<-datas$y
print("Data before preprocessing;row(gene),col(patients+3)")
print(dim(RawToy))
RawToy<-RawToy[RawToy$Gene_Symbol!="",]
RawToy<-RawToy[!duplicated(RawToy[,2]),]

# change the non log 2 scaled data to log2 scale
maxs = sapply( RawToy[, c(-1,-2,-3)], max ) 
not_log2_scale_ids = names( which(maxs > 100 ) )

for( j in 1 : length(not_log2_scale_ids ) ) {
  exception = RawToy[,not_log2_scale_ids[j]]<1
  RawToy[exception,not_log2_scale_ids[j]] = 1
  
  temp = log2( RawToy[,not_log2_scale_ids[j] ] )
  RawToy[,not_log2_scale_ids[j] ] = temp
  
}


#summary(RawToy)

Toy1000<- NormalizeToy(RawToy)
Toy1000<-GetVar(Toy1000)

temp1<-Toy1000[,c(1,2,3)]
temp2<-round(Toy1000[,c(-1,-2,-3)],digits = 3)
Toy1000<-cbind(temp1,temp2)
print("Data Preprocessed; row(genes), col(patients +4)")
print("remove duplicated genes, no name genes, log2 scale, normalize, round to digit 3")
print(dim(Toy1000))
Toy1000<-Toy1000[rev(order(Toy1000$VAR)),]
#Toy1000<-Toy1000[1:6000,]
Toy1000<-Toy1000[sample(length(Toy1000[,1]),6000),]


r_name<-as.character(Toy1000[,2])
data<-Toy1000[,c(-1,-2,-3,-length(Toy1000))]
data<-as.data.frame(t(data))
colnames(data)<-r_name
#remove no name patients

data$index<-sample(1:5,dim(data)[1],replace = TRUE)
data$result<-CancerResult[,2]
data<-data[substr(row.names(data),1,3)=="GSM",]



print("Cancer Result ratio")
table(data$result)
print("Number of Data;after feature selection;row(patients),col(gene+index+result)")
print(dim(data))


write.csv(data,"FinalData_GSM_gene_index_result_random.csv",row.names = TRUE)

