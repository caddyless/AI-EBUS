#library(ROCit)
library(pROC)

file_path = "E:\\OneDrive - sjtu.edu.cn\\Submission\\Medicasl-Video\\raw-data\\Perspective-Result.csv"

data = read.csv(file_path)

y_true = as.integer(data$BM)

mode = c('Fusion', 'B', 'F', 'E')
for (i in 1:4){
  m = mode[i]
  key = paste0(m, '.score')
  y_score = data[[key]]
  #rocit_emp = rocit(y_score, y_true, method='emp')
  #auc = ciAUC(rocit_emp)
  myroc = roc(y_true, y_score)
  ci.auc(myroc)
  #print(paste(m, ' ', as.character(auc$AUC), ' ', as.character(auc$lower), ' ', as.character(auc$upper)))
}


