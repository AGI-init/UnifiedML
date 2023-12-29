import torchvision

names = [
    'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
    'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
    'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
    'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
    'Caltech101', 'Caltech256', 'CelebA', 'WIDERFace', 'SBDataset',
    'USPS', 'Kinetics400', "Kinetics", 'HMDB51', 'UCF101',
    'Places365', 'Kitti', "INaturalist", "LFWPeople", "LFWPairs",
    'TinyImageNet',
    # 'Custom'
]

paths = [
    # 'null' if task == 'Custom' else
    f'World.Datasets.{task}.{task}' if task not in torchvision.datasets.__all__
    else f'torchvision.datasets.{task}' for task in names]

if __name__ == '__main__':
    out = ""
    for task, dataset_path in zip(names, paths):
        f = open(f"./{task.lower()}.yaml", "w")

        write = fr"""Env: World.Environments.Datums.Datums
Dataset: {dataset_path}
metric: 
    accuracy: World.Metrics.Accuracy
env:
    dataset: ${{dataset}}
    test_dataset: ${{test_dataset}}
    low: {'null' if task == 'Custom' else 0}
    high: {'null' if task == 'Custom' else 1}
    batch_size: ${{batch_size}}
    num_workers: ${{num_workers}}
dataset:
    Transform: {'transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64)])' if task == 'CelebA' else 'null'}
logger:
    log_actions: ${{not:${{generate}}}}
suite_name: classify
task_name: {'${format:${Dataset}}' if task == 'Custom' else task}
discrete: true
train_steps: 200000
stddev_schedule: 'linear(1.0,0.1,100000)'
frame_stack: null
action_repeat: null
nstep: 0
evaluate_per_steps: 1000
evaluate_episodes: 1
learn_per_steps: 1
learn_steps_after: 0
seed_steps: 0
rand_steps: 0
log_per_episodes: 300
RL: false
online: false  # Same as offline: true
"""

        write = fr"""imports:
    - classify
Dataset: {dataset_path}
env:
    low: {'null' if task == 'Custom' else 0}
    high: {'null' if task == 'Custom' else 1}
dataset:
    Transform: {'transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64)])' if task == 'CelebA' else 'null'}
"""

        f.write(write)
        f.close()
        out += ' "' + task.lower() + '"'
    print(out)
