# load necessary packages
library(ggformula)
library(dplyr)
library(readr)
library(corrplot)
library(caret)
library(glmnet)
library(tidyverse)
library(VIM)
library(GGally)
library(ggplot2)
library(RColorBrewer)

# read in 04cars data
vehicles04 <- read_csv('04cars.csv')

# tell R to treat categorical predictors as factors, correct height to numeric
vehicles04 <- vehicles04 %>%
  mutate(across(c(Type, Sport, SUV, Wagon, Minivan, Pickup), factor),
         Height = as.numeric(Height))

### Dealing with Missing Values

# identify proportion of rows with missing values
( dim(vehicles04)[1] - dim(na.omit(vehicles04))[1] ) / dim(vehicles04)[1]

# check the proportion of missing values per column
aggr(vehicles04, numbers = TRUE, sortVars = TRUE, cex.axis = 0.7, gap = 3, 
     ylab = c('Proportion of Rows Missing Values', 'Missingness Pattern'))

### Look for a pattern of missingness

# check the proportion of rows with NAs for each vehicle
cars <- vehicles04[vehicles04$Pickup == 0,]
trucks <- vehicles04[vehicles04$Pickup == 1,]
# cars
dim(cars[!complete.cases(cars),])[1] / dim(cars)[1]
# trucks
dim(trucks[!complete.cases(trucks),])[1] / dim(trucks)[1]

# since every truck row is missing values, see which columns are missing
aggr(trucks, numbers = TRUE, sortVars = TRUE, cex.axis = 0.7, gap = 3, 
     ylab = c('Proportion of Truck Rows Missing Values', 'Missingness Pattern for Trucks'))

# Length and Height have NAs for every truck row
# remove these columns so this information is not lacking disproportionately for trucks
vehicles04 <- vehicles04 %>%
  dplyr::select(-Length, -Height) %>%
  # then remove rows with missing vals
  na.omit()

### Check Multicollinearity

# examine the correlations between the numeric predictors
vehicles04 %>%
  dplyr::select_if(is.numeric) %>%
  cor() %>%
  corrplot()

### Check for Skewness and Transform Variables

# look at the residual plots for a mlr model using all predictors
plot(lm(Retailprice ~ ., data = vehicles04))  # heteroscedasticity and non-normality

### Examine the distribution of the numeric response
gf_histogram(~ Retailprice, data = vehicles04, fill = 'red', alpha = 0.7) %>%
  gf_labs(title = 'Distribution of Retailprice')

# log transform the skewed numeric response variable and remove original
vehicles04 <- vehicles04 %>%
  mutate(log_Retailprice = log(Retailprice)) %>%
  dplyr::select(-Retailprice)

# check the number of classes for each categorical predictor
str(vehicles04)   # all factors have 3 or fewer levels

### Plausible Tuning Parameters (lambda)

# begin with a wide exponential range of lambda values
lambdalist <- exp((-200:100)/10)

# use 10-fold CV to test models with different lambda vals, see where optimal value is
set.seed(10)
ctrl <- trainControl(method = 'cv', number = 10)
elastic_net <- train(log_Retailprice ~ .,
                     data = vehicles04,
                     method = 'glmnet',
                     tuneGrid = expand.grid(alpha = seq(0, 1, 0.2), lambda = lambdalist),
                     trControl = ctrl,
                     metric = 'MAE')
# view the best tuning parameters
elastic_net$bestTune

# since the optimal lambda was quite small, we can reduce the lambda range
lambdalist <- seq(0.001, 1, 0.001)

### Fitting Models (Single CV)

# set up 10-fold CV
set.seed(10)
ctrl <- trainControl(method = 'cv', number = 10)

# fit robust regression models using 10-fold CV on full model
robust_fit <- train(log_Retailprice ~ .,
                data = vehicles04,
                method = 'rlm',
                trControl = ctrl,
                maxit = 100,
                metric = 'MAE')

# fit elastic net models using 10-fold CV on full model
elastic_net_fit <- train(log_Retailprice ~ .,
                     data = vehicles04,
                     method = 'glmnet',
                     tuneGrid = expand.grid(alpha = seq(0, 1, 0.2), lambda = lambdalist),
                     trControl = ctrl,
                     metric = 'MAE')

### Outer Cross-Validation

# set up 5-fold outer CV
set.seed(10)
n <- dim(vehicles04)[1]
groups <- rep(1:5, length = n)
cvgroups <- sample(groups, size = n)

# set up storage for outer test predictions
all_outer_preds <- rep(NA, length = n)

# initiate outer CV
for(ii in 1:5){
  # split outer test and train groups
  in_outer_test <- (cvgroups == ii)
  outer_test <- vehicles04 %>%
    filter(in_outer_test)
  dataused <- vehicles04 %>%
    filter(!in_outer_test)
  
  # use dataused and inner 10-fold CV for fitting the models
  # robust regression
  robust <- train(log_Retailprice ~ .,
                  data = dataused,
                  method = 'rlm',
                  trControl = ctrl,
                  maxit = 100,
                  metric = 'MAE')
  
  # elastic net
  lambdalist <- seq(0.001, 1, 0.001)
  elastic_net <- train(log_Retailprice ~ .,
                       data = dataused,
                       method = 'glmnet',
                       tuneGrid = expand.grid(alpha = seq(0, 1, 0.2), lambda = lambdalist),
                       trControl = ctrl,
                       metric = 'MAE')
  
  # grab the best models from each
  best_mods <- list(robust, elastic_net)
  
  # grab the MAE corresponding to the best model for each
  lowest_MAEs <- c(min(robust$results$MAE), min(elastic_net$results$MAE))
  
  # choose the best model of the two
  best_mod <- best_mods[[which.min(lowest_MAEs)]]
  
  # use the best model to make predictions on the outer test set
  outer_preds <- best_mod %>%
    predict(outer_test)
  
  # store the predictions in the appropriate location
  all_outer_preds[in_outer_test] <- outer_preds
}

# calculate the outer cross-validated MAE on the exponentiated response (for interpretability)
outer_MAE <- mean(abs(exp(all_outer_preds) - exp(vehicles04$log_Retailprice)))
outer_MAE

# calculate the outer cross-validated MSE on the exponentiated response (for interpretability)
outer_MSE <- mean((exp(all_outer_preds) - exp(vehicles04$log_Retailprice)) ^ 2)
outer_MSE

### Overall Best Model

# best robust regression model
robust_fit$results[which.min(robust_fit$results$MAE),]

# best elastic net model
elastic_net_fit$results[which.min(elastic_net_fit$results$MAE),]

# fit the final model on the full dataset
# elastic net model with alpha = 0.2 and lambda = 0.003 has lowest MAE
final_model <- elastic_net_fit$finalModel

# plot the MAE of the elastic net models against the values of lambda and alpha
elastic_net_fit$results %>%
  dplyr::select(MAE, lambda, alpha) %>%
  ggplot(aes(lambda, alpha, color = MAE)) +
  geom_tile() +
  scale_color_gradient(low = 'yellow', high = 'red') +
  labs(title = 'MAE heat map for elastic net tuning parameters')

### Two Most Important Predictors

# grab the variable importance dataframe
var_imp_df <- varImp(final_model)

# convert the rownames to their own column
var_imp_df <- var_imp_df %>%
  rownames_to_column(var = 'Variable')

# reset the column names
colnames(var_imp_df) <- c('Variable', 'Importance')

# order by variable importance
var_imp_df <- var_imp_df[order(var_imp_df$Importance, decreasing = TRUE),]

# reset the row names
rownames(var_imp_df) <- c(1:14)

# generate a bar graph of variable importance    #### HOW CAN I ORDER THE COLUMNS IN DESCENDING ORDER???
var_imp_df[1:5,] %>%
  gf_col(Importance ~ Variable, fill = 'red', alpha = 0.6) %>%
  gf_labs(title = 'Five most important predictors in the model')

# plot each of the two most important variables against the predicted response

# append the predicted response to the df
vehicles_copy <- vehicles04
vehicles_copy <- vehicles_copy %>%
  mutate(predicted_response = predict(elastic_net_fit, vehicles04))

# generate the graphs
par(mfrow = c(2,1))
vehicles_copy %>%
  gf_violin(predicted_response ~ Pickup, fill =~ Pickup) %>%
  gf_labs(title = 'Highest predicted log retail prices are for cars',
          x = 'Pickup Truck',
          y = 'Predicted Log Retail Price')
vehicles_copy %>%
  gf_violin(predicted_response ~ SUV, fill =~ SUV) %>%
  gf_labs(title = 'Highest predicted log retail prices are not SUVs',
          x = 'SUV',
          y = 'Predicted Log Retail Price')