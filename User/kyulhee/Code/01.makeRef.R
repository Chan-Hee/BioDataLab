#################################### Hgnc symbol reference file ####################################

setwd("C:\\test")  
file_temp <- file("1.counts", open="r")
raw_ref <- read.table(file_temp, header=F)
#read TCGA expression sample file
names(raw_ref) <- c("ids","counts")
raw_ref$ids <- as.character(raw_ref$ids)
id_sets <- strsplit(raw_ref$ids, ".", fixed = TRUE)
# "." often used for different meaning. 'fixed' option could improve this matter.
ids <- NULL
for(i in id_sets){
  ids <- c(ids, unlist(i)[1])
}
#'id_sets' has ids and versions(behind the "."), extract ids only.


hgnc_syms <- read.csv("HGNC-Ensembl.csv", header=TRUE)
#file from website. here: https://www.genenames.org/cgi-bin/download?col=gd_hgnc_id&col=md_ensembl_id&status=Approved&status=Entry+Withdrawn&status_opt=2&where=%28%28gd_pub_chrom_map+not+like+%27%25patch%25%27+and+gd_pub_chrom_map+not+like+%27%25alternate+reference+locus%25%27%29+or+gd_pub_chrom_map+IS+NULL%29+and+gd_locus_type+%3D+%27gene+with+protein+product%27&order_by=md_ensembl_id&format=text&limit=&hgnc_dbtag=on&submit=submit

ref <- subset(hgnc_syms, hgnc_syms$Ensembl.ID.supplied.by.Ensembl. %in% ids)
index <- NULL
for(j in ref$Ensembl.ID.supplied.by.Ensembl.){
  index <- c(index, which(ids == j))
}
ref <- cbind(ref,index)
write.csv(ref, "hgnc_symbols_ref.csv",row.names=FALSE)

# intersect with GEO
inter_GEO_TCGA <- GEOvsTCGA("C:\\test\\GSE9974_GPL570_1.txt", "C:\\test\\hgnc_symbols_ref.csv")
ref_intersect <- subset(ref, ref$Approved.Symbol %in% inter_GEO_TCGA)
write.csv(ref_intersect, "hgnc_symbols_ref_inter.csv",row.names=FALSE)

}


#################################### GEO vs TCGA ####################################
GEOvsTCGA <- function(geo_sam, TCGA_ref){
  TCGA_file <- read.csv(TCGA_ref)
  #TCGA_file <- read.csv("C:\\test\\hgnc_symbols_ref.csv")
  TCGA_genes <- as.character(TCGA_file$Approved.Symbol)
  GEO_sam_file <- file(geo_sam, open="r")
  #GEO_sam_file <- file("C:\\test\\GSE9974_GPL570_1.txt", open="r")
  GEO_genes <- as.character(read.table(GEO_sam_file, header=TRUE)$Gene_Symbol)
  
  TCGA_genes <- data.frame(a=unique(TCGA_genes))
  GEO_genes <- data.frame(a=unique(GEO_genes[which(GEO_genes!="")]))
  GEO_genes$a
  
  inter_GEO_TCGA <- intersect(TCGA_genes$a, GEO_genes$a)
  return(inter_GEO_TCGA)
}
