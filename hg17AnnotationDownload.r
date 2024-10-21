library('IlluminaHumanMethylation450kanno.ilmn12.hg19')
annotation = getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
library(data.table)

head(annotation)
fwrite(as.data.frame(annotation), 'hg17.450k.annotation.csv')
