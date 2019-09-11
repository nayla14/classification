#Homework Machine Learning 2 : Classification
#Author : Nayla Azmi Afifi

data <- read.csv("https://raw.githubusercontent.com/arikunco/machinelearning/master/dataset/HR_comma_sep.csv", 
                 sep=",", header = TRUE)
data$left <- as.factor(data$left)
summary(data)


# package yang memuat fungsi rpart (decision tree)
library(rpart) 
# package yang memuat fungsi randomForest
library(randomForest)

#split training: test, 75:25
## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))
## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]

# model with logistic regression
logitMod <- glm(left ~ satisfaction_level + last_evaluation  + number_project  + average_montly_hours
              + time_spend_company + factor(Work_accident)         + factor(promotion_last_5years)
              + factor(sales)         + factor(salary), data=train, family=binomial(link="logit"))
summary(logitMod)
pred <- predict(logitMod, data.frame(test),type="response")
log_predict <- ifelse(pred > 0.5,1,0)
log_predict

library(caret)
nrow(as.data.frame(log_predict))
nrow(as.data.frame(test$left))
a <- as.factor(log_predict)
b <- as.factor(test$left)
summary(a)
summary(b)
##install.packages('e1071', dependencies=TRUE)
confusionMatrix(data = a, reference = b)
#Accuracy : 0.7829 
#Sensitivity : 0.9237 = recall        
#Specificity : 0.3486 = precission

# model with dec tree
tree <- rpart(left ~ ., 
              data = data.frame(train), method = "class")

conf_decTree <- table(test[,'left'],predict(tree, data.frame(test),type="class"))
conf_decTree
acc_decTree <- (conf_decTree[1,1]+conf_decTree[2,2])/(conf_decTree[1,1]+conf_decTree[1,2]+conf_decTree[2,1]+conf_decTree[2,2])
acc_decTree #accuracy = 0.9672
rec_decTree <- conf_decTree[2,2]/(conf_decTree[2,1]+conf_decTree[2,2])
rec_decTree #recall = 0.916122
prec_decTree <- conf_decTree[2,2]/(conf_decTree[1,2]+conf_decTree[2,2])
prec_decTree #precission = 0.9481398


# model with randomForest
randomFor <- randomForest(left ~ ., data = data.frame(train), ntree=1000, importance = TRUE)

conf_randomforest <- table(test[,'left'],predict(randomFor, data.frame(test), type="class"))
conf_randomforest 
acc_randomforest <- (conf_randomforest[1,1]+conf_randomforest[2,2])/(conf_randomforest[1,1]+conf_randomforest[1,2]+conf_randomforest[2,1]+conf_randomforest[2,2])
acc_randomforest #accuracy = 0.9914667
rec_randomforest <- conf_randomforest[2,2]/(conf_randomforest[2,1]+conf_randomforest[2,2])
rec_randomforest #recall = 0.9727669
prec_randomforest <- conf_randomforest[2,2]/(conf_randomforest[1,2]+conf_randomforest[2,2])
prec_randomforest #precission = 0.9922222


### Conclusion :  Setelah menjalankan 3 metode yaitu Logistic Regression, Decission Tree, dan Random Forest,
###               model terbaik adalah model Random Forest dengan nilai accuracy, recall, dan precission terbesar.
