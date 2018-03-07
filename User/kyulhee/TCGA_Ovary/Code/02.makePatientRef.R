merge_pro <- merge_pro[,c(1,2)]

patient_id <- NULL

for(i in merge_pro$barcode){
  ch <- as.character(i)
  temp <- unlist(strsplit(ch,"-"))[c(1,2,3)]
  patient_id <- c(patient_id, paste(temp[1],temp[2],temp[3], sep = "-"))
}

merge_pro_ch <- cbind(merge_pro, patient_id)
write.csv(merge_pro_ch, "file_name-patient_ID.csv", row.names = FALSE)
