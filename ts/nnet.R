


# R version of load_planar_dataset() on https://datascience-enthusiast.com/DL/Planar-data-classification-with-one-hidden-layer.html

# Could use python code to export result of load_planar_dataset() as CSV:
# import pandas as pd
# df1 = pd.DataFrame(np.transpose(X), columns = ['X1','X2'])
# df2 = pd.DataFrame(np.transpose(Y), columns = ['Y'])
# df = pd.concat([df1, df2], axis = 1)
# df.to_csv('planar_flower.csv', index = False)

# Or create data in R as a data frame
planar_dataset <- function(){
  set.seed(1)
  m <- 400
  N <- m/2
  D <- 2
  X <- matrix(0, nrow = m, ncol = D)
  Y <- matrix(0, nrow = m, ncol = 1)
  a <- 4
  
  for(j in 0:1){
    ix <- seq((N*j)+1, N*(j+1))
    t <- seq(j*3.12,(j+1)*3.12,length.out = N) + rnorm(N, sd = 0.2)
    r <- a*sin(4*t) + rnorm(N, sd = 0.2)
    X[ix,1] <- r*sin(t)
    X[ix,2] <- r*cos(t)
    Y[ix,] <- j
  }
  
  d <- as.data.frame(cbind(X, Y))
  names(d) <- c('x1','x2','y')
  d
}



sigmoid <- function(x){
  1/(1+exp(-x))
}

deriv <- function(x){
  fx = sigmoid(x)
  fx*(1-fx)
}

mse <- function(y, prd){
  mean((y - prd)^2)

}
forward <- function(i, data, w, b){
  h1 = w[1] * data[i,1] + w[2] * data[i,2] + b[1]
  h1. = sigmoid(h1)
  h2 = w[3] * data[i,1] + w[4] * data[i,2] + b[2]
  h2. = sigmoid(h2)
  prd = w[5] * h1. + w[6] * h2. + b[3]
  prd. = sigmoid(prd)
  
  list(h1 = h1, h2 = h2, prd = prd, prd. = prd.)
}


# --- Calculate partial derivatives.
backward <- function(i ,y, data , forward, w, b , lr){

  x1 = data$x1[i]
  x2 = data$x2[i]

  list2env(forward, env = environment())

  # partial deriv 
  L_prd = -2 * (y[i] - prd)

  # Neuron o1
  prd_w5 = h1 * deriv(prd)
  prd_w6 = h2 * deriv(prd)
  prd_b3 = deriv(prd)

  prd_h1 = w[5] * deriv(prd)
  prd_h2 = w[6] * deriv(prd)

  # Neuron h1
  h1_w1 = x1 * deriv(h1)
  h1_w2 = x2 * deriv(h1)
  h1_b1 = deriv(h1)

  # Neuron h2
  h2_w3 = x1 * deriv(h2)
  h2_w4 = x2 * deriv(h2)
  h2_b2 = deriv(h2)

  
  # Update
  # Neuron h1
  w1 = w[1] - lr * L_prd * prd_h1 * h1_w1
  w2 = w[2] - lr * L_prd * prd_h1 * h1_w2
  b1 = b[1] - lr * L_prd * prd_h1 * h1_b1

  # Neuron h2
  w3 = w[3] - lr * L_prd * prd_h2 * h2_w3
  w4 = w[4] - lr * L_prd * prd_h2 * h2_w4
  b2 = b[2] - lr * L_prd * prd_h2 * h2_b2

  # Neuron o1
  w5 = w[5] -lr * L_prd * prd_w5
  w6 = w[6] - lr * L_prd * prd_w6
  b3 = b[3] - lr * L_prd * prd_b3
  

  # Return
  update <- list (
    w = c(
      w1,
      w2,
      w3,
      w4,
      w5,
      w6),
    b = c(
      b2,
      b1,
      b3)
  )

  update

}

rescale <- function(x){
  m = mean(x)
  s= sd(x)
  (x-m)/s

}
######################################3

.data <- planar_dataset()

#library(ggplot2)
ggplot(.data, aes(x = x1, y = x2, color = factor(y))) +
  geom_point()

y <- .data$y
data <- data.frame(x1 = .data$x1, x2 = .data$x2)

w = rnorm(6)
b = rnorm(3)
lr = .01



go <- forward(i = 1,data, w, b)

n <- length(y)
prd <- numeric(n)
epoch <- 100
loss <- numeric(epoch)

for ( j in seq_len(epoch)){
  
  for (i in 1:n){

    back <- backward (
      i = 1, 
      data = data , 
      y = y, 
      forward = go, 
      w = w, 
      b = b , 
      lr = lr)
    w <- back$w
    b <- back$b
    
    go <- forward(i = 1,data, w, b)
    prd[i] <- go$prd.
    
    
  }

  loss[j] <- mse(y, prd)
  if (j %% 10 == 0) cat(j, '\n')

}



plot(1:epoch, loss)

#plot(y , prd)

table(y, ifelse(prd>0, 1, 0))
