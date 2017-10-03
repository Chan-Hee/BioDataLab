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
    if(nrow(Ndata)==nrow(data) && all(data$Gene_Symbol==Ndata$Gene_Symbol)){
      data<-cbind(data,Ndata[,c(-1,-2,-3)])
      NGSM<-colnames(Ndata[4:length(Ndata)])
      Ndiagnose<-rep(as.integer(substr(filenames[i],nchar(filenames[i])-4,nchar(filenames[i])-4),times=length(NGSM)))
      Ncancer<-data.frame(GSM_NUMBER=NGSM,CANCER=Ndiagnose)
      cancer<-rbind(cancer,Ncancer)
    }

    i=i+1
  }
  
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
data<-MergeToyFile(1000,"/Users/chanhee/Google Drive/RA/DATA/cancer_normal_database/GEO_GPL570")

RawToy<-data$x
RawToy<-RawToy[RawToy$Gene_Symbol!="",]
RawToy<-RawToy[!duplicated(RawToy[,2]),]

CancerResult<-data$y


Toy1000<- NormalizeToy(RawToy)
Toy1000<-GetVar(Toy1000)
temp1<-Toy1000[,c(1,2,3)]
temp2<-round(Toy1000[,c(-1,-2,-3)],digits = 3)
Toy1000<-cbind(temp1,temp2)

total<-nrow(Toy1000)

ToyVar1<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.01))
ToyVar2<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.02))
ToyVar5<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.05))
ToyVar10<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.1))
ToyVar20<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.2))
ToyVar30<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.3))

write.csv(Toy1000,"Toy1000.csv",row.names = FALSE)
write.csv(ToyVar1,"ToyVar1.csv",row.names = FALSE)
write.csv(ToyVar2,"ToyVar2.csv",row.names = FALSE)
write.csv(ToyVar5,"ToyVar5.csv",row.names = FALSE)
write.csv(ToyVar10,"ToyVar10.csv",row.names = FALSE)
write.csv(ToyVar20,"ToyVar20.csv",row.names = FALSE)
write.csv(ToyVar30,"ToyVar30.csv",row.names = FALSE)
write.csv(CancerResult,"CancerResult.csv",row.names = FALSE)
