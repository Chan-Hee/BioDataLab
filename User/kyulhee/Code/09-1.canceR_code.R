setwd("C:\\test")

source_code <- read.csv("Tissue_source_code.csv", header=TRUE)
TCGA_trns <- read.csv("TCGA_GEO_source.csv", header=TRUE)

trns <- NULL
for(i in 1:length(source_code$Study.Name)){
  trns <- rbind(trns, TCGA_trns[TCGA_trns[,1] == source_code$Study.Name[i],c(2,3)])
}

source_code <- cbind(source_code[,c(1,3)],trns)

write.csv(source_code, "TCGA_GEO_pro.csv", row.names = FALSE)
