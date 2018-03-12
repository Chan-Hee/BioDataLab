setwd("C:\\test\\sam")
large_table <- read.csv("TCGA_genes_FPKM-UQ_1.csv", header = TRUE)
GEO_6000 <- names(read.csv("C:\\test\\FinalData_GSM_gene_index_result.csv", header = TRUE))
GEO_6000_genes <- GEO_6000[2:length(GEO_6000)-2]
p <- GEO_6000_genes[-1]
TCGA_whole <- names(large_table)
TCGA_whole_gene <- TCGA_whole[-1]
q <- TCGA_whole_gene[1:(length(TCGA_whole_gene)-3)]

inter_6000 <- intersect(p,q)
names(inter_6000) <- "intersect"

write.csv(inter_6000, "inter_GEO_TCGA_geneset.csv",row.names = FALSE)
