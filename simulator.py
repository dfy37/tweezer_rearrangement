import gym
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import json

class AtomRearrangeEnv(gym.Env):
    def __init__(self, size=5, array=None):
        super().__init__()
        self.size = size
        self.max_steps = 20
        self.reset(array=array)
        plt.ion()

    def reset(self, p=0.5, array=None):
        self.steps = 0
        if array is not None:
            if isinstance(array, list):
                array = np.array(array)
            self.array = array
            return self.array.copy().flatten()
        
        self.array = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                if np.random.rand() < p:
                    self.array[i, j] = 1

        return self.array.copy().flatten()

    def step(self, x1, y1, y2, dire):
        reward = -1  # 每步惩罚
        done = False
        array = deepcopy(self.array)

        assert len(y1)==len(y2)
        if dire == 0:
            moving_tweezer = array[x1,:][:,y1]
            for i in range(len(x1)):
                array[x1[i],y1] = 0
            for i in range(len(x1)):
                array[x1[i],y2] += moving_tweezer[i,:]
        elif dire == 1:
            moving_tweezer = array[y1,:][:,x1]
            for i in range(len(x1)):
                array[y1,x1[i]] = 0
            for i in range(len(x1)):
                array[y2,x1[i]] += moving_tweezer[:,i]
        array = np.mod(array, 2)

        self.array = array
        self.steps += 1
        print(sum(self.array.flatten()))

        return array.copy().flatten(), reward, done, {}

    def render(self, name='Matrix View'):
        fig, ax = plt.subplots()
        ax.imshow(self.array, cmap='viridis', interpolation='none')
        ax.set_title(name)
        plt.colorbar(ax.images[0], ax=ax)

        # 框出中间的 3x3 区域
        if self.size >= 3:
            start = (self.size - 10) // 2
            end = start + 10
            # 绘制红色矩形框（边界外侧）
            rect = plt.Rectangle((start - 0.5, start - 0.5), 10, 10, linewidth=2,
                                edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        plt.pause(0.3)
        
    @property
    def action_space(self):
        return gym.spaces.Discrete(2 * self.size * self.size)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(self.size * self.size,), dtype=np.int32)


if __name__ == '__main__':
    np.random.seed(42)

    data = json.load(open('/Users/duanfeiyu/Documents/AtomRL/PathJson/Path_16391.json', 'r'))

    simulator = AtomRearrangeEnv(size=16, array=data[0]['state'])
    simulator.render()

    actions = []
    for x in data:
        action = (x['operation']['fixed_indices'], x['operation']['select1'], x['operation']['select2'], 1 - x['operation']['axis'])
        actions.append(action)

    for i, action in enumerate(actions):
        simulator.step(*action)
        simulator.render(name = f"Step {i}")