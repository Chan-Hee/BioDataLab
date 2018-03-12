setwd("C:\\test\\site")

file_list <- list.files(pattern = "*.csv")
samples <- NULL
types <- NULL
for(i in file_list){
  site_csv <- read.csv(i, header = FALSE)
  type <- unlist(strsplit(i,".", fixed = TRUE))[1]
  samples <- c(samples, as.character(site_csv$V1))
  types <- c(types, rep(type, length(site_csv$V1)))
}

sites <- data.frame(samples, types)
write.csv(sites, "Sites.csv",row.names = FALSE)
