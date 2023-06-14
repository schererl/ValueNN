from Amazons import Amazons
from agentes import StochasticModel, NoModel, ValueNetwork
from metamodel import METAMODEL
import stats

import os
import json
import time


stats.generate_heatmaps()





## FIRST GOOD TEST
### HERE I COULD FORCE A FAIR OUTCOME OF MODEL X MODEL WHEN USING STOCHASTIC AGENT
# rnd = NoModel()
# model = METAMODEL(need_train=True)
# model.train_models(rnd, rnd, 10000)
# model.test_model(rnd, 100)
# model.agent_model = StochasticModel(model.agent_model, 0.3)
# model.test_model(rnd, 100, p2=True)
# model.test_model(model.agent_model, 100, p2=True)

#rnd = NoModel()
#model = METAMODEL(need_train=False, model_name='TOPMODEL')
#weak_model = METAMODEL(need_train=False, model_name='m')
# model.test_model(rnd, 150)
# model.test_model(rnd, 150, p2=True)
#weak_model.agent_model = StochasticModel(weak_model.agent_model, 0.3)
#model.agent_model = StochasticModel(model.agent_model, 0.3)
#model.test_model(weak_model.agent_model, 150)
#model.test_model(weak_model.agent_model, 150, p2=True)

#model.store_model('TOPMODEL')
#file_name = 'checkTOPMODEL.json'
#store_states(model.agent_model, 1000, file_name)
#print_probabilities(compute_probabilities(file_name))

# random vs random test
#rnd = NoModel()
#store_states(rnd, 1000, 'ST-rnd')
#print_probabilities(compute_probabilities('ST-rnd'))

# model vs model test
#model = METAMODEL(need_train=False, model_name='TOPMODEL')
#model.agent_model = StochasticModel(model.agent_model, 0.3)
#store_states(model.agent_model, 1000, 'ST-topmodel')
#print_probabilities(compute_probabilities('ST-topmodel'))









# TESTING MODEL S
# rnd = NoModel()
# model = METAMODEL(need_train=False, model_name='m')
# modelS = METAMODEL(need_train=False, model_name='mS')


# output=''
# output += 'valueNN x RND: '+modelS.test_model(rnd, 100)+'\n'
# output += 'RND x valueNN: ' + modelS.test_model(rnd, 100, p2=True, str_conc=output)+'\n'
# output += 'valueNN x NaiveNN: '+ modelS.test_model(model.trained_model, 100, str_conc=output)+'\n'
# output += 'NaiveNN x valueNN: ' + model.test_model(modelS.trained_model, 100, str_conc=output)+'\n'
# output += 'NaiveNN x NaiveNN: ' + model.test_model(model.trained_model, 100, str_conc=output)+'\n'
# output += 'valueNN x valueNN: ' + modelS.test_model(modelS.trained_model, 100, str_conc=output)+'\n'
#os.system('clear')
#print(output)

# file_name = 'checkBias.json'
# store_states(model.trained_model, 10000, file_name)
# print_probabilities(compute_probabilities(file_name))
# file_name = 'checkRandom.json'
# store_states(rnd, 10000, file_name)
# print_probabilities(compute_probabilities(file_name))


# rnd = NoModel()
# model = METAMODEL(need_train=False, model_name='mS')

# rnd = NoModel()
# model = METAMODEL(need_train=True)
# model.train(rnd, rnd, 1000)
# model.store_model('modelTrain')

# model.test_model(rnd, 100)
# model.test_model(rnd, 100, p2=True)
# model.test_model(model.agent_model, 100)

#model.test_model(model.trained_model, 100)


    
