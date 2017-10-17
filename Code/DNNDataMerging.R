################################################### Merging Files #########################################

SelectFile<-function(filenames){

  randVector=sample(1:length(filenames),replace = FALSE)
  return(filenames[randVector])
  
}

MergeUntil<-function(filenames,n){
  
  
  data<-read.table(filenames[1],header=TRUE)
  GSM<-colnames(data[4:length(data)])
  diagnose<-rep(as.integer(substr(filenames[1],nchar(filenames[1])-4,nchar(filenames[1])-4),times=length(GSM)))
  cancer<-data.frame(GSM_NUMBER=GSM,CANCER=diagnose)
  i=2
  
  while(ncol(data)<=n+3){
    
    Ndata<-read.table(filenames[i],header=TRUE)
    
    #print (paste(n, " : ", filenames[i], " dim: " , dim(data)[1], ", ", dim(data)[2], " na: ", sum(is.na(data)) , " i: " , i , sep="") )
    
    #print( paste("is new data contain na? count of nas in the Ndata: ", sum(is.na(Ndata)) ) )
    
    if(nrow(Ndata)==nrow(data) && all(data$Gene_Symbol==Ndata$Gene_Symbol)){
      data<-cbind(data,Ndata[,c(-1,-2,-3)])
      NGSM<-colnames(Ndata[4:length(Ndata)])
      Ndiagnose<-rep(as.integer(substr(filenames[i],nchar(filenames[i])-4,nchar(filenames[i])-4),times=length(NGSM)))
      Ncancer<-data.frame(GSM_NUMBER=NGSM,CANCER=Ndiagnose)
      cancer<-rbind(cancer,Ncancer)
    }
    
    i=i+1
    #drop sample which contains at least one NA
    data[ , colSums(is.na(data)) == 0]
    
    
  }
  
  print( c(n, " : " , sum(is.na(data)) ) )
  
  return(list(x=data,y=cancer))
}



MergeToyFile<-function(n,mypath){
  
  filenames<-list.files(path = mypath,full.names = TRUE)
  filenames<-SelectFile(filenames)
  datalist <- MergeUntil(filenames,n)
  
  return(datalist)
}
################################################### Normalization #########################################

NormalizeToy<-function(RawToy){
  temp<-RawToy[,c(1,2,3)]
  for(i in colnames(RawToy)[4:length(RawToy)]){
    NormalizedV<-(RawToy[,i]-mean(RawToy[,i]))/sd(RawToy[,i])
    RawToy[,i]<-NormalizedV
  }
  RawToy[,1]<-temp[,1]
  RawToy[,2]<-temp[,2]
  RawToy[,3]<-temp[,3]
  
  
  return(RawToy)
}
#################################################### VarianceTest ##########################################
GetVar<-function(Toy1000){
  Toy1000$VAR<-apply(Toy1000[,4:length(Toy1000)],1,sd)
    return(Toy1000)
}

#################################################### Main ##################################################

set.seed(2017)
data<-MergeToyFile(100,"/Users/chanhee/Google Drive/RA/DATA/cancer_normal_database/GEO_GPL570")

RawToy<-data$x
RawToy<-RawToy[RawToy$Gene_Symbol!="",]
RawToy<-RawToy[!duplicated(RawToy[,2]),]

CancerResult<-data$y


Toy1000<- NormalizeToy(RawToy)
Toy1000<-GetVar(Toy1000)
temp1<-Toy1000[,c(1,2,3)]
temp2<-round(Toy1000[,c(-1,-2,-3)],digits = 3)
Toy1000<-cbind(temp1,temp2)

write.csv(Toy1000,"DNN10000.csv",row.names = FALSE)
write.csv(CancerResult,"CancerResult.csv",row.names = FALSE)
