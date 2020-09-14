getwd()
rm(list = ls())

#loading data file
df <- read.csv("train_cab.csv", header=T, na.strings=c("","NA"))

head(df)

x = c("ggplot2", "corrgram", "Metrics", "rlang", "DMwR","DataExplorer", "DataCombine","caret", "randomForest", "e1071","geosphere",
      "DataCombine", "pROC", "doSNOW", "class", "readxl","ROSE","dplyr", "plyr", "reshape","xlsx",
      "pbapply", "unbalanced", "dummies", "MASS" , "gbm" ,"Information", "rpart", "tidyr", "miscTools")


# #install.packages if not
#lapply(x, install.packages)

# #load libraries
lapply(x, require, character.only = TRUE)
#install.packages("Metrics")
library("Metrics")
summary(df)
str(df)

df$pickup_datetime <- gsub('\\ UTC','',df$pickup_datetime)

#Splitting Date and time
df$Date <- as.Date(df$pickup_datetime)
df$Year <- substr(as.character(df$Date),1,4)
df$Month <- substr(as.character(df$Date),6,7)
df$Weekday <- weekdays(as.POSIXct(df$Date), abbreviate = F)
df$Date <- substr(as.character(df$Date),9,10)
df$Time <- substr(as.factor(df$pickup_datetime),12,13)
df$fare_amount <- as.numeric(as.character(df$fare_amount))


############################# 
#Function to calculate distance based on lat and long of 2 location points. 

lat1 = df['pickup_latitude']
lat2 = df['dropoff_latitude']
long1 = df['pickup_longitude']
long2 = df['dropoff_longitude']

gcd.hf <- function(lon1, lat1, lon2, lat2){
  rad <- pi/180
  a1 <- lat1 * rad
  a2 <- lon1 * rad
  b1 <- lat2 * rad
  b2 <- lon2 * rad
  dlon <- b2 - a2
  dlat <- b1 - a1
  a <- (sin(dlat/2))^2 + cos(a1) * cos(b1) * (sin(dlon/2))^2
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  R <- 6378.145
  d <- R * c
  return(d)
}

#Running the function for all rows in dataframe
for (i in 1:nrow(df))
{
  df$distance[i]= gcd.hf(df$pickup_longitude[i], df$pickup_latitude[i], df$dropoff_longitude[i], df$dropoff_latitude[i])
}


#Now we can drop the columns for latitude/longitude as we have new column- Distance
df = subset(df, select = -c(pickup_datetime ,pickup_latitude,dropoff_latitude,pickup_longitude,dropoff_longitude))

summary(df)

#########################################################################
# Checking Missing data #
#########################################################################

plot_missing(df)
apply(df, 2, function(x) {sum(is.na(x))}) # in R, 1 = Row & 2 = Col
#Creating dataframe with missing values present in each variable
null_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
null_val$Columns = row.names(null_val)

names(null_val)[1] = "null_percentage"

#Calculating percentage missing value
null_val$null_percentage = (null_val$null_percentage/nrow(df)) * 100
# Sorting null_val in Descending order
null_val = null_val[order(-null_val$null_percentage),]
row.names(null_val) = NULL
# Reordering columns
null_val = null_val[,c(2,1)]
#viewing the % of missing data for all variales
null_val

#We have seen that null values are very less in our data set i.e. less than 1%.
df <- na.omit(df)
#Verifying missing values after deletion
sum(is.na(df))
names(df)


#structure of data or data types
str(df)
#Summary of data
summary(df)
#unique value of each count
apply(df, 2,function(x) length(table(x)))


##########################################################################################

#Outlier detection
#We check the outliers for the continuous variables (passenger count, distance and fare)
summary(df)
boxplot(df$fare_amount)
boxplot(df$passenger_count)
boxplot(df$distance)

boxplot(df$fare_amount)$out
boxplot(df$passenger_count)$out
boxplot(df$distance)$out

count(df$fare_amount > 100)
count(df$passenger_count > 6)
count(df$distance > 150)


# Removing outliers :
#We have seen that fare_amount has negative values which should be removed
df$fare_amount[df$fare_amount<=0] <- NA
df$fare_amount[df$fare_amount>500] <- NA
sum(is.na(df))

###removing passangers count more than 6
df$passenger_count[df$passenger_count<1] <- NA
df$passenger_count[df$passenger_count>6] <- NA
df$passenger_count[df$passenger_count==1.3] <- NA
sum(is.na(df))
summary(df$passenger_count)

###removing distance more than 200

df$distance[df$distance>200] <- NA
sum(is.na(df))
plot_missing(df)

df <- na.omit(df)
sum(is.na(df))
head(df)
summary(df)

# Categorizing data in 2 categories "continuous" and "categorical"
#Fare_amount being our target variable is excluded from the list.
cont = c('distance')
cata = c('Weekday', 'Month',  'Time', 'Date' , 'passenger_count')

#########################################################################
# Visualizing the data #
#########################################################################

#library(ggplot2)

#Plot fare amount Vs. time
ggplot(df, aes(x=factor(Time), y=fare_amount)) + stat_summary(fun="mean", geom="bar") + labs(title = "Mean Fare Amount Vs. Time")

#Plot fare amount Vs. the date
ggplot(df, aes(x=factor(Date), y=fare_amount)) + stat_summary(fun="mean", geom="bar") + labs(title = "Mean Fare Amount Vs. Date")

#Plot fare amount Vs. the days of the week
ggplot(df, aes(x=factor(Weekday), y=fare_amount)) + stat_summary(fun="mean", geom="bar")+ labs(title = "Mean Fare Amount Vs. weekday")

#Plot Fare amount Vs. months
ggplot(df, aes(x=factor(Month), y=fare_amount)) + stat_summary(fun="mean", geom="bar") + labs(title = "Mean Fare Amount Vs. Month")

#Plot Fare amount Vs. Year
ggplot(df, aes(x=factor(Year), y=fare_amount)) + stat_summary(fun="mean", geom="bar") + labs(title = "Mean Fare Amount Vs. Year")

#Plot Fare amount Vs. passenger_count 
ggplot(df, aes(x=factor(passenger_count), y=fare_amount)) + stat_summary(fun="mean", geom="bar") + labs(title = "Mean Fare Amount Vs. passenger_count")


#Plot fare amount Vs. distance
plot(x = df$distance ,y = df$fare_amount,
     xlab = "distance",
     ylab = "fare_amount",
     main = "fare_amount vs distance"
)
cor(df$fare_amount, df$distance)


################################################################
# Feature Selection #
################################################################
## Dimension Reduction
#We will remove Month, weekday, passenger_count and Date column also as it is not required
names(df)
df = subset(df, select = -c(Weekday, Date, Month, passenger_count))

################################################################
# Feature Scaling #
################################################################

##We skip normalization as scale difference of distance and fare_amount is not that significant Normalization.
head(df)
df2 <- df
#Creating dummy variables for categorical variables
library(mlr)
df1 = dummy.data.frame(df, cata)

#Viewing data after adding dummies
head(df1)


#Creating 2 parts of the dataset (df1) into train and test set.
require(caTools)
set.seed(101)
sample = sample.split(df1$fare_amount, SplitRatio = .75)
train = subset(df1, sample == TRUE)
test  = subset(df1, sample == FALSE)

######################################################

#Training the linear regression model : 

lrmodel <- lm(fare_amount ~ ., data = train)

summary(lrmodel)

#Predict for test set
test$y_pred_lm = predict(lrmodel, newdata = test[,-1])

mse_lm <- mean((test$fare_amount - test$y_pred_lm)^2)
RMSE_lm <- sqrt(mse_lm)

##################################################################

# Decision Tree regression : 

fit = rpart(fare_amount ~ ., data = train,  
            control=rpart.control(minsplit=25))


#Predict for test cases
test$y_pred_dt = predict(fit, test[, -1])
mse_dt <- mean((test$fare_amount - test$y_pred_dt)^2)
RMSE_dt <-sqrt(mse_dt)

###########################################################

#Random Forest Regression

library(randomForest)
set.seed(1234)
sum(is.na(df))
regressor = randomForest(x = train[,-1],
                         y = train$fare_amount,
                         ntree = 1000, 
                         importance = TRUE)

# Predicting a new result for test set with Random Forest Regression

test$y_pred_rf = predict(regressor, test[,-1])
mse_rf <- mean((test$fare_amount - test$y_pred_rf)^2)
RMSE_rf <- sqrt(mse_rf)

#############################################################################################
#SVM Regression

library(e1071)
regressor = svm(formula = fare_amount ~ .,
                data = train, kernel = 'radial',
                cost = 10)

summary(regressor)

# Predicting result in test set

test$y_pred_svm = predict(regressor, test[,-1])

mse_svm <- mean((test$fare_amount - test$y_pred_svm)^2)
RMSE_svm <- sqrt(mse_svm)


###############################################################################################
library(e1071)
#install.packages("rlang")
library(caret)

#Gradient Boosted Machine Regression

sample = sample.split(df2$fare_amount, SplitRatio = .75)
train = subset(df2, sample == TRUE)
test  = subset(df2, sample == FALSE)

# load the package
library(gbm)
# fit model
str(train)
test$Year <- as.factor(test$Year)
test$Time <- as.factor(test$Time)
train$Time <- as.factor(train$Time)
train$Year <- as.numeric(train$Year)
test$Year <- as.numeric(test$Year)
str(train)
str(test)
gbm.fit <- gbm(
  formula = fare_amount ~ .,
  distribution = 'gaussian',
  data = train,
  n.trees = 10000,
  interaction.depth = 1,
  shrinkage = 0.01,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

test$y_pred_gb = predict.gbm(gbm.fit, test, n.trees = 1000)

mse_gb <- mean((test$fare_amount - test$y_pred_gb)^2)
RMSE_gb <- sqrt(mse_gb)

str(test)

df <- read.csv("test.csv")

df$pickup_datetime <- gsub('\\ UTC','',df$pickup_datetime)

#Splitting Date and time
df$Date <- as.Date(df$pickup_datetime)
df$Year <- substr(as.character(df$Date),1,4)
df$Month <- substr(as.character(df$Date),6,7)
df$Weekday <- weekdays(as.POSIXct(df$Date), abbreviate = F)
df$Date <- substr(as.character(df$Date),9,10)
df$Time <- substr(as.factor(df$pickup_datetime),12,13)

str(df)

lat1 = df['pickup_latitude']
lat2 = df['dropoff_latitude']
long1 = df['pickup_longitude']
long2 = df['dropoff_longitude']

gcd.hf <- function(lon1, lat1, lon2, lat2){
  rad <- pi/180
  a1 <- lat1 * rad
  a2 <- lon1 * rad
  b1 <- lat2 * rad
  b2 <- lon2 * rad
  dlon <- b2 - a2
  dlat <- b1 - a1
  a <- (sin(dlat/2))^2 + cos(a1) * cos(b1) * (sin(dlon/2))^2
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  R <- 6378.145
  d <- R * c
  return(d)
}

#Running the function for all rows in dataframe
for (i in 1:nrow(df))
{
  df$distance[i]= gcd.hf(df$pickup_longitude[i], df$pickup_latitude[i], df$dropoff_longitude[i], df$dropoff_latitude[i])
}

#Now we can drop the columns for latitude/longitude as we have new column- Distance
df = subset(df, select = -c(pickup_datetime ,pickup_latitude,dropoff_latitude,pickup_longitude,dropoff_longitude))
df = subset(df, select = -c(Weekday, Date, Month, passenger_count))
df$Year <- as.numeric(df$Year)
df$predicted_values <- predict.gbm(gbm.fit, df, n.trees = 1000)
str(df)
write.csv(df, "Result.csv")
head(df)
