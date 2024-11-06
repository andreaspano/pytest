source('./ts/nnet2.R') 

layer_size <- getLayerSize(trn_x, trn_y, hidden_neurons = 4)
#layer_size

init_params <- initializeParameters(trn_x, layer_size)
#lapply(init_params, dim)

fwd_prop <- forwardPropagation(trn_x, init_params, layer_size)
#lapply(fwd_prop, dim)

cost <- computeCost(trn_x, trn_y, fwd_prop)
#cost


back_prop <- backwardPropagation(trn_x, trn_y, fwd_prop, init_params, layer_size)
#lapply(back_prop, function(x) dim(x))

update_params <- updateParameters(back_prop, init_params, learning_rate = 0.01)
#lapply(update_params, function(x) dim(x))

EPOCHS = 60000
HIDDEN_NEURONS = 40
LEARNING_RATE = 0.1

train_model <- trainModel(
  trn_x, 
  trn_y, 
  hidden_neurons = HIDDEN_NEURONS, 
  num_iteration = EPOCHS, 
  lr = LEARNING_RATE)



plot(1:EPOCHS, train_model$cost_hist, type = 'l')



prd_y <- makePrediction(tst_x, tst_y, HIDDEN_NEURONS)
prd_y <- round(prd_y)

table(tst_y, prd_y)
