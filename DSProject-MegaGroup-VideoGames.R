# imports 
library(tidyverse) 
library(tidytext)
library(ggplot2)
library(moments)
library(e1071)
library(caret)
library(lattice)
library(gplots)
df <- read.csv("vgsales.csv")
options(repr.plot.width = 40, repr.plot.height = 15)

# display data 
head(df)
summary(df)
colnames(df)
unique(df$Genre)
sum(is.na(df))
sum(is.null(df))
nrow(df)

# visuals
# na sales throughout different platforms  
ggplot(df, aes(Genre, NA_Sales)) +
  geom_bar(stat="identity", fill="red") +
  labs(title="Bar Plot", x="X-axis", y="Y-axis")
# na sales throughout different genres   
ggplot(df, aes(Platform, NA_Sales)) +
  geom_bar(stat="identity", fill="blue") +
  labs(title="Bar Plot", x="X-axis", y="Y-axis")

# feature selection 
df <- df[c("Genre","Platform", "NA_Sales","EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales")]
head(df)

# removing outliers 
  # NA Sales
Q1 <- quantile(df[["NA_Sales"]], 0.25)
Q3 <- quantile(df[["NA_Sales"]], 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

outliers <- df[["NA_Sales"]] < lower_bound | df[["NA_Sales"]] > upper_bound
df <- df[!outliers, ]

  # EU Sales
Q1 <- quantile(df[["EU_Sales"]], 0.25)
Q3 <- quantile(df[["EU_Sales"]], 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

outliers <- df[["EU_Sales"]] < lower_bound | df[["EU_Sales"]] > upper_bound
df <- df[!outliers, ]

  # JP Sales
Q1 <- quantile(df[["JP_Sales"]], 0.25)
Q3 <- quantile(df[["JP_Sales"]], 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

outliers <- df[["JP_Sales"]] < lower_bound | df[["JP_Sales"]] > upper_bound
df <- df[!outliers, ]

  # Other Sales
Q1 <- quantile(df[["Other_Sales"]], 0.25)
Q3 <- quantile(df[["Other_Sales"]], 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

outliers <- df[["Other_Sales"]] < lower_bound | df[["Other_Sales"]] > upper_bound
df <- df[!outliers, ]

# checking for skweness 
skewness(df[["NA_Sales"]])
skewness(df[["EU_Sales"]])
skewness(df[["JP_Sales"]])
skewness(df[["Other_Sales"]])

# smoothing the noise in sales values  
  # Define a function to calculate the simple moving average
calculate_sma <- function(data, window_size) {
  ma <- stats::filter(data, rep(1/window_size, window_size), sides = 2)
  return(c(rep(NA, window_size-1), ma))
}
  # Specify the window size for the moving average
window_size <- 3

  # Apply the moving average to the specified column
  # NA Sales
df$NA_Smoothed_Value <- calculate_sma(df[["NA_Sales"]], window_size)[1:nrow(df)]
df[["NA_Smoothed_Value"]]
df$EU_Smoothed_Value <- calculate_sma(df[["EU_Sales"]], window_size)[1:nrow(df)]
df[["EU_Smoothed_Value"]]
df$JP_Smoothed_Value <- calculate_sma(df[["JP_Sales"]], window_size)[1:nrow(df)]
df[["JP_Smoothed_Value"]]
df$Other_Smoothed_Value <- calculate_sma(df[["Other_Sales"]], window_size)[1:nrow(df)]
df[["Other_Smoothed_Value"]]

# sum null values were introduced here and we are removing them
sum(is.na(df))
sum(is.null(df))
df <- na.omit(df)
# check for skewness again 
skewness(df[["NA_Smoothed_Value"]])
skewness(df[["EU_Smoothed_Value"]])
skewness(df[["JP_Smoothed_Value"]])
skewness(df[["Other_Smoothed_Value"]])

# add regions for classification 
df$Region <- apply(df[, c("NA_Smoothed_Value", "EU_Smoothed_Value", "JP_Smoothed_Value","Other_Smoothed_Value")], 1, function(x) {
  max_index <- which.max(x)
  if (max_index == 1) {
    return(paste("North America"))
  } else if (max_index == 2) {
    return(paste("Europe"))
  } else if (max_index == 3) {
    return(paste("Japan"))
  } else {
    return(paste("other"))
  }
})

tail(df)

# encoding nominal classes 
df$Encoded_Genre <- as.integer(factor(df$Genre))
df$Encoded_Platform <- as.integer(factor(df$Platform))

df <- df[,c("Encoded_Genre", "Encoded_Platform", "NA_Smoothed_Value", "EU_Smoothed_Value", "JP_Smoothed_Value", "Other_Smoothed_Value", "Region")]

# Splitting the target and input features 
x <- df[,c("Encoded_Genre", "Encoded_Platform", "NA_Smoothed_Value", "EU_Smoothed_Value", "JP_Smoothed_Value", "Other_Smoothed_Value")]
y <- df$Region

# seed for reproducibilty
set.seed(456)

# test and train parts 
split_index <- createDataPartition(y, p = 0.7, list = FALSE)
train_set <- df[split_index, ]
test_set <- df[-split_index, ]

# training
# SVM
svm_model <- svm(as.factor(Region) ~ ., data = train_set, kernel = "linear")

# predict SVM
predictions <- predict(svm_model, newdata = test_set[, c("Encoded_Genre", "Encoded_Platform", "NA_Smoothed_Value", "EU_Smoothed_Value", "JP_Smoothed_Value", "Other_Smoothed_Value")])

#confusion matrix for SVM 
confusion_matrix <- table(predictions, test_set$Region)
confusion_matrix

confusion_df <- as.data.frame(as.table(confusion_matrix))
  # Create a heatmap using ggplot2 SVM
ggplot(confusion_df, aes(x = Var2, y = predictions, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  labs(title = "Confusion Matrix Heatmap", x = "Actual", y = "Predicted")

# accuracy for SVM
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy * 100))


# training Naive Bayes
nb_model <- naiveBayes(as.factor(Region) ~ ., data = train_set)

# predict Naive Bayes
predictions_nb <- predict(nb_model, newdata = test_set[, c("Encoded_Genre", "Encoded_Platform", "NA_Smoothed_Value", "EU_Smoothed_Value", "JP_Smoothed_Value", "Other_Smoothed_Value")])

#confusion matrix for Naive Bayes
confusion_matrix_nb <- table(predictions_nb, test_set$Region)
confusion_matrix_nb

confusion_df_nb <- as.data.frame(as.table(confusion_matrix_nb))
# Create a heatmap using ggplot2
ggplot(confusion_df_nb, aes(x = Var2, y = predictions_nb, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  labs(title = "Confusion Matrix Heatmap", x = "Actual", y = "Predicted")

# accuracy for Naive Bayes
accuracy_nb <- sum(diag(confusion_matrix_nb)) / sum(confusion_matrix_nb)
print(paste("Accuracy:", accuracy_nb * 100))