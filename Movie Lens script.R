#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,] 

# Make sure userId and movieId in validation set are also in edx set
#validation set 

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###################################################################################
#################################################

#Save edx and validation files
save(edx, file="edx.RData")
save(validation, file = "validation.RData")

#Loading library 
library(tidyverse)
library(ggplot2)
library(dplyr)
library(markdown)
library(knitr)
library(caret)
### Exploratory analysis

#first we explore the data 
str(edx)
str(validation)

# check for NA value 
anyNA(edx)
# number of movies and users in data set 
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# let's look for the most  ratings
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))  

#number of rating per movies 
edx %>% count(movieId) %>% ggplot(aes(n))+
  geom_histogram(color = "black" , fill= "light blue")+
  scale_x_log10()+
  ggtitle("Rating Number Per Movie")+
  theme_gray()
#number of rating per user
edx %>% count(userId) %>% ggplot(aes(n))+
  geom_histogram(color = "black" , fill= "light blue")+
  ggtitle(" Number of Rating Per User")+
  scale_x_log10()+
  theme_gray()

#number of rating for each movie genres
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>% ggplot(aes(genres,count)) + 
  geom_bar(aes(fill =genres),stat = "identity")+ 
  labs(title = " Number of Rating for Each Genre")+
  theme(axis.text.x  = element_text(angle= 90, vjust = 50 ))+
  theme_light()
 
#Partition the data 
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
#to make sure we don???t include users and movies in the test set that do not appear in 
#the training set, we remove these entries using the semi_join function:
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
  
#RMSE calculation Function 

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}

# first model 
Mu_1 <- mean(train_set$rating)
Mu_1

naive_rmse <- RMSE( Mu_1,test_set$rating)
naive_rmse

#creating a table for the RMSE result to store all the result from each method to compare

rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

#second model 
#as we saw on the exploratory analysis some are rated more than other 
#We can augment our previous model by adding the term  b_i 
#to represent average ranking for movie i
#Y_{u,i} = \mu + b_i + \varepsilon_{u,i}
#We can again use least squared to estimate the movie effect 
Mu_2 <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - Mu_2))
movie_avgs
#we can see that variability in the estimate as plotted here 
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))
#let's see how the prediction improves 
predicted_ratings <- Mu_2 + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",  
                                     RMSE = model_1_rmse))
rmse_results 
                   
# Third model 
# let;s compure the user I for , for those who ratedover 100 movies 
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")
#hat there is substantial variability across users ratings as well. T
#his implies that a further improvement to our 
#model may be:
#$Y~u,i~ = ?? + b~i~ + ??~u,i~$
#we could fit this model by using use the lm() function but as mentioned earlier it 
#would be very slow
#lm(rating ~ as.factor(movieId) + as.factor(userId))
#so here is the code 
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - Mu_2 - b_i))
user_avgs

#now let's see how RMSE improved this time 
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = Mu_2 + b_i + b_u) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_3_rmse))
rmse_results

## RMSE of the validation set

valid_pred_rating <- validation %>%
  left_join(movie_avgs, by = "movieId" ) %>% 
  left_join(user_avgs , by = "userId") %>%
  mutate(pred = Mu_2 + b_i + b_u ) %>%
  pull(pred)

model_3_valid <- RMSE(validation$rating, valid_pred_rating)
model_3_valid
rmse_results <- bind_rows( rmse_results, 
                           data_frame(Method = "Validation Results" , RMSE = model_3_valid))
rmse_results

