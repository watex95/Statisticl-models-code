#install packages and dependencies
#install.packages("devtools", dependencies = TRUE)

library(devtools)
devtools::install_github("ayhandis/creditR")
library(creditR)


#Load and clean data
# Attaching the library
library(creditR)

#Model data and data structure
data("germancredit")
str(germancredit)

#Preparing a sample data set
sample_data <- germancredit[,c("duration.in.month","credit.amount",
        "installment.rate.in.percentage.of.disposable.income",
        "age.in.years","creditability")]

#Converting the ‘Creditability’ (default flag) variable into numeric type
sample_data$creditability <- ifelse(sample_data$creditability == "bad",1,0)

#Calculating the missing ratios
missing_ratio(sample_data)

#Data wrangling
#Splitting the data into train and test sets
traintest <- train_test_split(sample_data,123,0.70)
train <- traintest$train
test <- traintest$test


#Applying WOE transformation on the variables
woerules <- woe.binning(df = train,target.var = "creditability",pred.var = 
                          train,event.class = 1)
train_woe <- woe.binning.deploy(train, woerules, add.woe.or.dum.var='woe')

#Creating a dataset with the transformed variables and default flag
train_woe <- woe.get.clear.data(train_woe,default_flag = "creditability",
                                prefix = "woe")

#Applying the WOE rules used on the train data to the test data
test_woe <- woe.binning.deploy(test, woerules, add.woe.or.dum.var='woe')

test_woe <- woe.get.clear.data(test_woe,default_flag = "creditability",prefix = "woe")



#Feature engineering

#Performing the IV and Gini calculations for the whole data set
IV.calc.data(train_woe,"creditability")
Gini.univariate.data(train_woe,"creditability")


#Creating a new dataset by Gini elimination. IV elimination is also possible
eliminated_data <- Gini_elimination(train_woe,"creditability",0.10)
str(eliminated_data)

#A demonstration of the functions useful in performing Clustering
clustering_data <- variable.clustering(eliminated_data,"creditability", 2)
clustering_data

# Returns the data for variables that have the maximum gini value in
#the dataset
selected_data <- variable.clustering.gini(eliminated_data,"creditability", 2)


correlation.cluster(eliminated_data,clustering_data,
                    variables = "variable",clusters = "Group")



# Model building and training
####################################################################################

#Creating a logistic regression model of the data
model= glm(formula = creditability ~ ., family = binomial(link = "logit"),
           data = eliminated_data)
summary(model)

#Calculating variable weights
woe.glm.feature.importance(eliminated_data,model,"creditability")

#Generating the PD values for the train and test data
ms_train_data <- cbind(eliminated_data,model$fitted.values)
ms_test_data <- cbind(test_woe[,colnames(eliminated_data)], 
                predict(model,type = "response", newdata = test_woe))
colnames(ms_train_data) <- c("woe.duration.in.month.binned",
      "woe.age.in.years.binned","woe.installment.rate.in.percentage.of.disposable.
      income.binned","creditability","PD")
colnames(ms_test_data) <- c("woe.duration.in.month.binned","woe.age.in.
      years.binned","woe.installment.rate.in.percentage.of.disposable.income.binned",
                            "creditability","PD")


#An example application of the Regression calibration method. The model is 
#calibrated to the test_woe data
regression_calibration <- regression.calibration(model,test_woe,"creditability")
regression_calibration$calibration_data
regression_calibration$calibration_model
regression_calibration$calibration_formula

#Creating a master scale
master_scale <- master.scale(ms_train_data,"creditability","PD")
master_scale


#Calibrating the master scale and the modeling data to the default rate of 5% 
#using the bayesian calibration method
ms_train_data$Score = log(ms_train_data$PD/(1-ms_train_data$PD)) 
ms_test_data$Score = log(ms_test_data$PD/(1-ms_test_data$PD)) 
bayesian_method <- bayesian.calibration(data = master_scale,average_score ="Score",
    total_observations = "Total.Observations",PD = "PD",central_tendency = 0.05,
    calibration_data = ms_train_data,calibration_data_score ="Score")

#After calibration, the information and data related to the calibration process can be obtained as follows
bayesian_method$Calibration.model
bayesian_method$Calibration.formula



#The Scaled score can be created using the following function
scaled.score(bayesian_method$calibration_data, "calibrated_pd", 3000, 15)


# Model evaluation and testing

#Calculating the Vif values of the variables.
vif.calc(model)

#Calculating the Gini for the model
Gini(model$fitted.values,ms_train_data$creditability)

#Performing the 5 Fold cross validation
k.fold.cross.validation.glm(ms_train_data,"creditability",5,1)



