import pandas as pd
import numpy as np
testcount = 10

###189*198###
file_directory = "/home/tjahn/Data/"
geneset = pd.read_csv(file_directory+"geneset.csv")
genedata = pd.read_csv(file_directory+"FinalData_GSM_gene_index_result.csv")



###Making Matrix###
matrix = np.zeros((testcount, len(geneset), 198))
###Each GeneSet###
geneset['gene_symbols'][188].split()



for i in range(testcount):
    for j in range(len(geneset)):
        gene_symbols = geneset['gene_symbols'][j].split(",")
        cnt = 0 
        for k in range(len(gene_symbols)):
            if gene_symbols[k] in genedata.columns :
                matrix[i][j][cnt] = genedata[gene_symbols[k]][i]
                cnt+=1
    print(matrix[i])    
#     pyplot.show()
