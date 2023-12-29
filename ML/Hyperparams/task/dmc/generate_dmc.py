import glob
import os

easy = ['cartpole_balance', 'cartpole_balance_sparse', 'cartpole_swingup', 'cup_catch', 'finger_spin', 'hopper_stand', 'pendulum_swingup', 'walker_stand', 'walker_walk']
medium = ['acrobot_swingup', 'cartpole_swingup_sparse', 'cheetah_run', 'finger_turn_easy', 'finger_turn_hard', 'hopper_hop', 'quadruped_run', 'quadruped_walk', 'reach_duplo', 'reacher_easy', 'reacher_hard', 'walker_run']
hard = ['humanoid_stand', 'reacher_hard', 'humanoid_walk', 'humanoid_run', 'finger_turn_hard']

if __name__ == '__main__':
    files = glob.glob(os.getcwd() + "/*")

    # Prints tasks by difficulty
    # print([f.split('.')[-2].split('/')[-1] for f in files if 'hard' in open(f, 'r').read() and 'generate' not in f])

    out = ""
    for task in easy + medium + hard:
        f = open(f"./{task.lower()}.yaml", "w")

        write = fr"""Env: World.Environments.DMC.DMC
suite_name: dmc
env:
    task: {task}
task_name: {task}
discrete: false
action_repeat: 2
frame_stack: 3
nstep: {1 if 'walker' in task else 3}
train_steps: {500000 if task in easy else 1500000 if task in medium else 15000000}
stddev_schedule: 'linear(1.0,0.1,{100000 if task in easy else 500000 if task in medium else 2000000})'
hd_capacity: 1e6
{'lr: 8e-5' if 'humanoid' in task else ''}
{'trunk_dim: 100' if 'humanoid' in task else ''}
{'batch_size: 512' if 'walker' in task else ''}
"""
        write = fr"""imports:
    - dmc
nstep: {1 if 'walker' in task else 3}
train_steps: {500000 if task in easy else 1500000 if task in medium else 15000000}
stddev_schedule: 'linear(1.0,0.1,{100000 if task in easy else 500000 if task in medium else 2000000})'
{'lr: 8e-5' if 'humanoid' in task else ''}
{'trunk_dim: 100' if 'humanoid' in task else ''}
{'batch_size: 512' if 'walker' in task else ''}
"""

        f.write(write)
        f.close()
        out += ' "' + task.lower() + '"'
    print(out)
