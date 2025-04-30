import random
import numpy as np

def default_task_selector(task_properties: dict, epsilon: float, alpha: float):
    preferred_task = random.choice(list(task_properties.keys())) # pick preferred task at random
    worst_criterion = np.inf # minimal criterion -> highest selection probability
    # print(task_properties) # uncomment to see probabilities computed for each task
    for key in task_properties:
        game_dict = task_properties[key]
        
        # incorporate how well the model is estimated to perform on task:
        if game_dict["n_episodes"] > 0:
            mean_reward = game_dict["total_reward"] / game_dict["n_episodes"]
        else:
            mean_reward = 0
        if "target_reward" in game_dict:
            criterion = (mean_reward - game_dict["target_reward"]) / game_dict["target_reward"]
        else:
            criterion = mean_reward

        # find how well the task has been explored:
        criterion += alpha * game_dict["n_episodes"]

        if criterion < worst_criterion:
            worst_criterion = criterion
            preferred_task = key

    N = len(task_properties)
    task_probabilities = dict()
    for key in task_properties:
        if key == preferred_task:
            task_probabilities[key] = epsilon / N + (1 - epsilon)
        else:
            task_probabilities[key] = epsilon / N
    return task_probabilities

def enhanced_task_selector(task_properties: dict, epsilon: float, alpha: float):
    totalprogress = 0
    totalnotargetprogress = None
    task_probabilities = dict()
    Nnontarget = 0
    for key in task_properties:
        if task_properties[key]['training_progress'] > 0:
            progress = task_properties[key]['training_progress']
        else:
            progress = 0
        totalprogress += 1 - progress/task_properties[key]['record']
        if progress<task_properties[key]['target_reward']:
            Nnontarget += 1
            if not totalnotargetprogress:
                totalnotargetprogress = 1-progress/task_properties[key]['target_reward']
            else:
                totalnotargetprogress += 1-progress/task_properties[key]['target_reward']
    if Nnontarget == 0:
        epsilon = 1
    for key in task_properties:
        if task_properties[key]['training_progress'] > 0:
            progress = task_properties[key]['training_progress']
        else:
            progress = 0
        task_probabilities[key] = epsilon*(1-progress/task_properties[key]['record'])/totalprogress
        if task_properties[key]['training_progress']<task_properties[key]['target_reward']: task_probabilities[key] += (1-epsilon)*(1-task_properties[key]['training_progress']/task_properties[key]['target_reward'])/totalnotargetprogress
    return task_probabilities