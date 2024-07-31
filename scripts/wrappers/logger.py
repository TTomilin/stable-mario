import gymnasium
import numpy as np
import wandb

class BaseRewardLogger(gymnasium.Wrapper):
    """Base class for loggers that track something about the reward gained in past n episodes"""

    def __init__(self, env: gymnasium.Env, episode_limit: int) -> None:
        super().__init__(env)
        self._current_episode_rewards = []
        self._episode_rewards = []
        self._episode_count = 0
        self._episode_limit = episode_limit

    def _updateEpisodeRewards(self, reward, truncated, terminated):
        "Updates rewards of current episodes and the rewards of the n episodes being tracked"
        self._current_episode_rewards.append(reward) # append current reward
        
        if terminated or truncated: # check if episode ended
            self._episode_rewards.append(sum(self._current_episode_rewards)) # add current episode reward to list
            self._current_episode_rewards = [] # empty current episode rewards
            self._episode_count += 1 # increment step count

    def step(self, action):
        return NotImplementedError

class LogVariance(BaseRewardLogger):
    """
    Logs the reward variance over n last episodes to wandb in a separate graph
    """

    def __init__(self, env: gymnasium.Env, episode_limit: int) -> None:
        super().__init__(env, episode_limit)
        wandb.define_metric(name="variance", step_metric="global_step")

    def step(self, action):
        """Take environment step, but also tracks variance"""
        observation, reward, terminated, truncated, info = self.env.step(action)

        self._updateEpisodeRewards(reward, truncated, terminated) # update the episode rewards

        if self._episode_count >= self._episode_limit:
            var = np.array(self._episode_rewards).var(ddof=1) # compute variance in episode rewards
            wandb.log({"variance": var}) # log variance
            self._episode_rewards = [] # empty list of episode rewards
            self._episode_count = 0 # reset episode counter
            
        return observation, reward, terminated, truncated, info
    
class LogRewardSummary(BaseRewardLogger):
    """
    Logs the minimum and maximum, and mean reward over n past episodes to wandb in a separate graph
    """

    def __init__(self, env: gymnasium.Env, episode_limit: int, with_wandb: bool) -> None:
        super().__init__(env, episode_limit)
        self.with_wandb = with_wandb
        if self.with_wandb:
            wandb.define_metric(name="Minimum Reward", step_metric="global_step")
            wandb.define_metric(name="Maximum Reward", step_metric="global_step")
            wandb.define_metric(name="Average Reward", step_metric="global_step")


    def step(self, action):
        """Take environment step, but also tracks minimum and maximum reward"""
        observation, reward, terminated, truncated, info = self.env.step(action)

        self._updateEpisodeRewards(reward, terminated, truncated) # update episode rewards
        if self._episode_count >= self._episode_limit:
            print(self._episode_rewards)
            reward_arr = np.array(self._episode_rewards)
            minReward = reward_arr.min() # retrieve smallest reward
            maxReward = reward_arr.max() # retrieve largest reward
            avgReward = reward_arr.mean() # retrieve average reward
            print(f"smallest reward past {str(self._episode_limit)} episodes: {minReward}")
            print(f"average reward past {str(self._episode_limit)} episodes: {avgReward}")
            print(f"largest reward past {str(self._episode_limit)} episodes: {maxReward}")
            if self.with_wandb:
                wandb.log({"Minimum Reward": minReward, "Average Reward": avgReward, "Maximum Reward": maxReward}) # log values
            self._episode_rewards = [] # empty list of episode rewards
            self._episode_count = 0 # reset episode counter
            
        return observation, reward, terminated, truncated, info
    
class ResumeLogger(gymnasium.Wrapper):
    """
    Logs the episode reward mean and episode length mean for resumed runs.
    """

    def __init__(self, env: gymnasium.Env, episode_limit: int) -> None:
        super().__init__(env, episode_limit)

    def step(self, action):
        """Take environment step, but also tracks minimum and maximum reward"""
        observation, reward, terminated, truncated, info = self.env.step(action)

        self._updateEpisodeRewards(reward, terminated, truncated) # update episode rewards

        if self._episode_count >= self._episode_limit:
            print(self._episode_rewards)
            reward_arr = np.array(self._episode_rewards)
            minReward = reward_arr.min() # retrieve smallest reward
            maxReward = reward_arr.max() # retrieve largest reward
            avgReward = reward_arr.mean() # retrieve average reward
            print(f"smallest reward past {str(self._episode_limit)} episodes: {minReward}")
            print(f"average reward past {str(self._episode_limit)} episodes: {avgReward}")
            print(f"largest reward past {str(self._episode_limit)} episodes: {maxReward}")
            wandb.log({"Minimum Reward": minReward, "Average Reward": avgReward, "Maximum Reward": maxReward}) # log values
            self._episode_rewards = [] # empty list of episode rewards
            self._episode_count = 0 # reset episode counter
            
        return observation, reward, terminated, truncated, info
    
class StepRewardLogger(gymnasium.Wrapper):
    """Wrapper that writes the rewards accumulated over each timestep to a text file"""

    def __init__(self, env: gymnasium.Env, log_dir: str, step_limit: int = 1000) -> None:
        super().__init__(env)
        self.line_limit = step_limit # default step limit corresponds to about 50 seconds of play
        self.reward_buffer = []
        self.log_dir = log_dir

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_buffer.append(reward) # append step's reward
        if len(self.reward_buffer) >= self.line_limit: # if sufficient rewards collected....
            with open(f"{self.log_dir}/step_rewards.txt", 'w', encoding='utf-8') as file:
                for reward in self.reward_buffer:
                    file.write(f"{reward}\n") # write them to file
            self.reward_buffer = [] # empty buffer

        return observation, reward, terminated, truncated, info # return required info