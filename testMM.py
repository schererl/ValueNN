from Amazons import Amazons
from train import ZEROMODEL, METAMODEL, NNModel, STOCHASTICNN, NOMODELUCT

DEBUG=False

# HERE: sModel2
#           training using standard backpropagation of outcomes and using complement in the evaluate model
#           model vs random model has advantage for both p1 and p2
#           model vs model model p1 has a huge advantage over p2

# HERE: sModel3
#           training using minimax backpropagation of outcomes and NOT using complement in the evaluate model
#           model vs rnd when mdl is p2 is very weak
#           model vs model model p2 has a huge advantage over p1

rnd = ZEROMODEL()

model = METAMODEL(need_train=True)
model.train(rnd, rnd, 10000)
model.store_model('testEliminations')
o1 = model.test_model(rnd, 100)
o2 = model.test_model(rnd, 100, p2=True, str_conc=o1)
model.test_model(model.trained_model, 1000, str_conc=o1+'\n'+o2)

# model = METAMODEL(need_train=False, model_name='pleasegodwork')
# model2 = METAMODEL(need_train=False, model_name='pleasegodwork')

# o1 = model.test_model(rnd, 100)
# o2 = model.test_model(rnd, 100, p2=True, str_conc=o1)
# model.test_model(model2.trained_model, 1000, str_conc=o1+'\n'+o2)


# if DEBUG:
#     model.train(rnd, rnd, 1000)
#     model.debug(rnd, 10)
# elif not model.trained_model:
#     print("running model from scratch")
#     model.train(rnd, rnd, 10000)
#     model.test_model(model.trained_model, 1000, p2=True)
#     model.store_model('sModel2')
# else: # training a new model with new experiences using NaiveNN as base
#     batch_size = 1000
#     it = 0
#     batch_it = 1
#     while(True):
#         model.trained_model = STOCHASTICNN(model, 0.75)
#         model.train(model.trained_model, rnd, batch_size, incremental=True, output = "batches: "+str(batch_it))
#         model.test_model(model.trained_model, 100)
#         it += batch_size
#         batch_it += 1
#         model.store_model("probModel")
