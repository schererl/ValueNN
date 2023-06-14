from Amazons import Amazons
from agentes import StochasticModel, NoModel, ValueNetwork
from metamodel import METAMODEL
import os
import json
import time

_stats_folder = 'STATISTICS/'
def store_states(model, n_games, filename):
    states = []

    for _ in range(n_games):
        game = Amazons()
        while not game.check_game_over():
            move = model.evaluate(game.deep_copy())
            game.play(*move)
            states.append(game.network_state.tolist())  # Convert ndarray to list here

    with open(_stats_folder+filename, 'w') as f:
        json.dump(states, f)


def compute_probabilities(filename):
    with open(_stats_folder+filename, 'r') as f:
        states = json.load(f)
    
    total_moves = len(states)
    ones_count = [0]*len(states[0])

    for state in states:
        for i, val in enumerate(state):
            if val == 1:
                ones_count[i] += 1

    probabilities = [count/total_moves for count in ones_count]

    return probabilities

def print_probabilities(lst):
    mover = lst[0:1]
    p1 = lst[1:26]
    p2 = lst[26:51]
    arrow = lst[51:]

    mover = ["{:.2f}".format(i) for i in mover]
    p1 = ["{:.2f}".format(i) for i in p1]
    p2 = ["{:.2f}".format(i) for i in p2]
    arrow = ["{:.2f}".format(i) for i in arrow]

    print(f"Mover {len(mover)}: {mover}")
    print(f"P1 {len(p1)}: {p1}")
    print(f"P2 {len(p2)}: {p2}")
    print(f"Arrow {len(arrow)}: {arrow}")

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

model = METAMODEL(need_train=False, model_name='TOPMODEL')
model.agent_model = StochasticModel(model, 0.3)
store_states(model.agent_model, 1000, 'ST-topmodel')
print_probabilities(compute_probabilities('ST-topmodel'))











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


    
