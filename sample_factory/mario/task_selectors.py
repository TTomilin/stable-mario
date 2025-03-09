import random
import numpy as np

def default_task_selector(task_properties: dict, epsilon: float, alpha: float):
    preferred_task = random.choice(list(task_properties.keys())) # pick preferred task at random
    worst_criterion = np.inf # minimal criterion -> highest selection probability

    for key, value in task_properties:
        game_dict = task_properties[key]
        
        # incorporate how well the model is estimated to perform on task:
        mean_reward = game_dict["total_reward"] / game_dict["n_episodes"]
        if "target_reward" in game_dict:
            criterion = mean_reward - game_dict["target_reward"]
        else:
            criterion = mean_reward

        # find how well the task has been explored:
        criterion += alpha * game_dict["n_episodes"]

        if criterion < worst_criterion:
            worst_criterion = criterion
            preferred_task = key

    p_rest = 1 - epsilon
    N = len(task_properties)
    task_probabilities = dict()
    for key, _ in task_properties:
        if key == preferred_task:
            task_probabilities[key] = p_rest / N + (1 - epsilon)
        else:
            task_probabilities[key] = p_rest / N
    print(task_probabilities)
    return task_probabilities