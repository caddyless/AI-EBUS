---
title: "R Notebook"
output: html_notebook
---

This is an [R Markdown](http://rmarkdown.rstudio.com) Notebook. When you execute code within the notebook, the results appear beneath the code. 

Try executing this chunk by clicking the *Run* button within the chunk or by placing your cursor inside it and pressing *Ctrl+Shift+Enter*. 

```{r}
library(pROC)
library(caret)

file_path = "E:\\OneDrive - sjtu.edu.cn\\Submission\\Medicasl-Video\\raw-data\\Perspective-Result.csv"

data = read.csv(file_path)

y_true = as.integer(data$BM)
mode = c('Fusion', 'B', 'F', 'E')

```


```{r}
for (i in 1:4){
  m = mode[i]
  key = paste0(m, '.score')
  y_score = data[[paste0(m, '.score')]]
  y_predict = data[[paste0(m, '.predict')]]
  myroc = roc(y_true, y_score)
  auc = ci.auc(myroc)
  out_content = paste(m, ' ', as.character(auc[2]), '(', as.character(auc[1]), ',', as.character(auc[3]), ')', sep='')
  
  cm = confusionMatrix(y_predict, y_true)
  
  print(out_content)
  # sprintf('Mode %s, AUC=%.4f(%.4f-%.4f)', m, auc$AUC, auc$lower, auc$upper)
}
```




Add a new chunk by clicking the *Insert Chunk* button on the toolbar or by pressing *Ctrl+Alt+I*.

When you save the notebook, an HTML file containing the code and output will be saved alongside it (click the *Preview* button or press *Ctrl+Shift+K* to preview the HTML file).

The preview shows you a rendered HTML copy of the contents of the editor. Consequently, unlike *Knit*, *Preview* does not run any R code chunks. Instead, the output of the chunk when it was last run in the editor is displayed.
