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
#################################################### CNNData ################################################
ConvertToCNN<-function(df){
  total<-nrow(df)-1
  sqrt<-as.integer(sqrt(total))+1
  square<-sqrt^2
  zeros<-as.data.frame(matrix(0,ncol = ncol(df) ,nrow =square-total-1 ))
  colnames(zeros)<-colnames(df)
  df<-rbind(df,zeros)
  return(df)
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

CNNToy1000<-ConvertToCNN(Toy1000)
total<-as.integer(nrow(CNNToy1000))

ToyVar1<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.01))
CNNToyVar1<-ConvertToCNN(ToyVar1)
ToyVar2<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.02))
CNNToyVar2<-ConvertToCNN(ToyVar2)
ToyVar5<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.05))
CNNToyVar5<-ConvertToCNN(ToyVar5)
ToyVar10<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.1))
CNNToyVar10<-ConvertToCNN(ToyVar10)
ToyVar20<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.2))
CNNToyVar20<-ConvertToCNN(ToyVar20)
ToyVar30<-tail(Toy1000[ordered(Toy1000$VAR),],n=as.integer(total*0.3))
CNNToyVar30<-ConvertToCNN(ToyVar30)
write.csv(CNNToy1000,"CNNToy1000.csv",row.names = FALSE)
write.csv(CNNToyVar1,"CNNToyVar1.csv",row.names = FALSE)
write.csv(CNNToyVar2,"CNNToyVar2.csv",row.names = FALSE)
write.csv(CNNToyVar5,"CNNToyVar5.csv",row.names = FALSE)
write.csv(CNNToyVar10,"CNNToyVar10.csv",row.names = FALSE)
write.csv(CNNToyVar20,"CNNToyVar20.csv",row.names = FALSE)
write.csv(CNNToyVar30,"CNNToyVar30.csv",row.names = FALSE)
write.csv(CancerResult,"CancerResult.csv",row.names = FALSE)
