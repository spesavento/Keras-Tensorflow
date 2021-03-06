# To upgrade my development environment, needed to do this: 
#Command line update $ sudo /usr/local/bin/pip install --upgrade virtualenv

# needed to update Rcpp 
install.packages("Rcpp")

#tensorflow is google's platform to experiment neural network
#keras a higher level library that runs tensorflow -- it simplifies tensoflow
install.packages("devtools")
require(devtools)

# Needed to update reticulate 
install_github("rstudio/reticulate")


devtools::install_github("rstudio/keras")

library(keras)
install_keras()
install_github("rstudio/tensorflow")
install_github("rstudio/keras")

#loading the keras inbuilt mnist dataset
data<-dataset_mnist() #downloads dataset

#separating train and test file
train_x<-data$train$x   #images ----- supervised
train_y<-data$train$y   #labels
test_x<-data$test$x
test_y<-data$test$y

rm(data)

# converting a 2D array into a 1D array for feeding into the MLP (multilayer perception) and normalising the matrix
#reducing dimensionality -- for the computer to be able to process 
# /255 gives everything a value from 0 to 1 (normalizing a matrix)
train_x <- array(train_x, dim = c(dim(train_x)[1], prod(dim(train_x)[-1]))) / 255
test_x <- array(test_x, dim = c(dim(test_x)[1], prod(dim(test_x)[-1]))) / 255

#converting the target variable to once hot encoded vectors using keras inbuilt function
train_y<-to_categorical(train_y,10)
test_y<-to_categorical(test_y,10)

#defining a keras sequential model
model <- keras_model_sequential()

#defining the model with 1 input layer[784 neurons], 1 hidden layer[784 neurons] with dropout rate 0.4 and 1 output layer[10 neurons]
#i.e number of digits from 0 to 9
#hyperparameters -- structuring 
model %>% 
  layer_dense(units = 784, input_shape = 784) %>% 
  layer_dropout(rate=0.4)%>%
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 10) %>% 
  layer_activation(activation = 'softmax')

#compiling the defined model with metric = accuracy and optimiser as adam.
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

#fitting the model on the training dataset
model %>% fit(train_x, train_y, epochs = 10, batch_size = 128)

#Evaluating model on the cross validation dataset
loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = 128)


#accuracy -- is it a 5 ? yes it's a 5?.