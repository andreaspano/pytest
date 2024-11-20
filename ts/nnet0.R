rm(list = ls(a = T))
require(ggplot2)
require(dplyr)
require(tidyr)
source('./ts/nnet4.R') 


data <- planar_dataset()

ggplot(data, aes(x = x1, y = x2, color = factor(y))) +
  geom_point()



dl <- prepare_data(data , y = 'y', xs = c('x1', 'x2'), frac = .8)

list2env(dl, globalenv())


#lapply(list(trn_x = trn_x, trn_y = trn_y, tst_x = tst_x, tst_y =  tst_y), dim)


#layer_size <- getLayerSize(trn_x, trn_y, hidden_neuron = 4)
#layer_size

#init_param <- initializeParameters(trn_x, layer_size)
#lapply(init_param, dim)

#fwd_prop <- forward_propagation(trn_x, init_param, layer_size)
#lapply(fwd_prop, dim)

#cost <- compute_cost(trn_x, trn_y, fwd_prop)
#cost


#back_prop <- backward_propagation(trn_x, trn_y, fwd_prop, init_param, layer_size)
#lapply(back_prop, function(x) dim(x))

#update_param <- updateParameters(back_prop, init_param, learning_rate = 0.01)
#lapply(update_param, function(x) dim(x))

EPOCHS = 90000
HIDDEN_NEURON = 4
LEARNING_RATE = 0.01

model <- trainModel(
  X = trn_x, 
  y = trn_y, 
  hidden_neuron = HIDDEN_NEURON, 
  epochs = EPOCHS, 
  lr = LEARNING_RATE)


data_epoch <- tibble(
  epoch = 1:EPOCHS, 
  cost_hist = model$cost_hist
)

ggplot(data_epoch) + 
  geom_line(aes(epoch, cost_hist), col = 'red')




prd_y <- make_prediction(X = tst_x, y = tst_y, hidden_neuron = HIDDEN_NEURON, param = model$param)
prd_y <- round(prd_y)

table(tst_y, prd_y)

prd <- tst |> 
  mutate(prd_y = as.vector(prd_y))

ggplot(prd, aes(x = x1, y = x2, color = factor(y))) +
  geom_point()


n <- 50
s <-  seq(-4, 4 , length.out = n)
base  <- expand_grid(x1 = s, x2 = s) |> 
  mutate(y = rep(0, n*n))
base_x <- base |> 
  select(x1, x2) |> 
  as.matrix() |> 
  t()

base_y <- base |> 
  pull(y) |> 
  t()

prd_base <-  make_prediction(X = base_x, y = base_y, hidden_neuron = HIDDEN_NEURON   , param = model$param)
prd_base <- round(prd_base)
base <- base |> 
  mutate(prd_base = as.vector(prd_base))

ggplot() +
  geom_point(aes(x = x1, y = x2, color = factor(prd_base)), data = base, shape = 1) +
  geom_point(aes(x = x1, y = x2, color = factor(prd_y)), data = tst, shape = 16, size = 4) + 
  geom_point(aes(x = x1, y = x2, color = factor(y)), data = trn, shape = 16, size = 2)

  
