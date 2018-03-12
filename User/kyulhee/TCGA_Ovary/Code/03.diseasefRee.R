setwd("/home/tjahn/GDC_Data/TCGA-High-grade_serous_ovarian_cancer/")
df <- read.csv("TCGA_genes_Ovary_FPKM.csv", header=TRUE)
ref <- read.csv("TCGA_disease_free.csv", header = TRUE)

d_free = NULL

for(i in df$patient_id){
  
  ele = ref[which(ref$track_name==i),]
  if(is.null(ele)){
    ele = c(i,"NOT_FINDED")
  }
  
  if(is.null(d_free)){
    d_free=ele
    start=1
  }else{
    d_free <- rbind(d_free, ele)
  }
}

write.csv(d_free, "patient-d_free.csv", row.names = FALSE)