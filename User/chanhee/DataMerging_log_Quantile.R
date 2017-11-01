# set working directory
# or adjust path to open the sample info file
samples = read.table(file="GPL570_sampleinfo.txt", header=T, sep="\t" ) 

# at the directory containing all GSE files
# get the list of file names starts with "GSE"
files = list.files(".", pattern="GSE")

# get the sample (GSM_ID) id list
BRCA_sample = samples
BRCA_sample_ids = as.vector( unique( BRCA_sample$GSM_ID ) )

# open file and 
# record if it contains samples from the cancer we want
# then save the sample's data into "collect" variable
collect = c()
for( i in 1:length(files) ){
  
  x = read.table(files[i], sep="\t", header=T)
  #initialize the data frame
  if(i == 1) {
    collect = x[1:54613,c(1:3)]
  }    
  
  tmp = data.frame( x[,which(colnames(x)%in%BRCA_sample_ids) ] )
  
  collect = cbind( collect, tmp[ 1:54613, ] ) 
  
  print( paste( i, " : " , dim(collect)[1], " " , dim(collect)[2]) );
  
  #cleaning up memory
  gc()
}


write.table(collect, file="GPL570.txt", sep="\t", quote=F, row.names=F)

data = collect
data = data[ , colSums(is.na(data)) == 0]

write.table(collect, file="GPL570_na_filttered.txt", sep="\t", quote=F, row.names=F)

# get maxs values of columns
maxs = sapply( data[, c(-1,-2,-3)], max ) 
not_log2_scale_ids = names( which(maxs > 100 ) )

# change the non log 2 scaled data to log2 scale
data_ = data
for( j in 1 : length(not_log2_scale_ids ) ) {
  
  data_[,not_log2_scale_ids[j] ] = log2( data_[,not_log2_scale_ids[j] ] )
  
}

# replace negative values to 0
# we do this because some cols negative by taking the log 2
data_[,c(-1,-2,-3)] <- as.data.frame(lapply(data_[,c(-1,-2,-3)], function(x){replace(x, x <0,0)}))

# cut the digits
data_ = format( data_, digits = 4)

# making sure if the changed digit is numeric
data_[, c(-1,-2,-3)] = sapply( data_[, c(-1,-2,-3)], as.numeric) 

# box plotting to see the distribution
boxplot( data_[, c(4:104) ] )

write.table(data_, file="GPL570_na_filttered_log2.txt", sep="\t", quote=F, row.names=F)

# install packages for normalization
source('http://bioconductor.org/biocLite.R')
biocLite('preprocessCore')
#load package
library(preprocessCore)

# normalize.quantile only accept matrix type
mat = as.matrix( data[,c(-1,-2,-3)] )

# exercise while it runs
post.norm <- normalize.quantiles(mat)

# see the difference before it is normlized
boxplot( post.norm[, c(1:100) ] )

data_n = data.frame( mat )
data_n = cbind( data_[,c(1:3)], data_n) 


write.table(data_n, file="GPL570_na_filttered_log2_normlized.txt", sep="\t", quote=F, row.names=F)