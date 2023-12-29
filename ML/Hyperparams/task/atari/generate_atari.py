# Full Atari ALE (108 games)
atari_tasks = [
    'Adventure', 'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis', 'Atlantis2',
    # 'Backgammon',
    'BankHeist', 'BasicMath', 'BattleZone', 'BeamRider', 'Berzerk', 'Blackjack', 'Bowling',
    'Boxing', 'Breakout', 'Carnival', 'Casino', 'Centipede', 'ChopperCommand', 'Combat', 'CrazyClimber',
    'Crossbow', 'Darkchambers', 'Defender', 'DemonAttack', 'DonkeyKong', 'DoubleDunk', 'Earthworld',
    'ElevatorAction', 'Enduro', 'Entombed', 'Et', 'FishingDerby', 'FlagCapture', 'Freeway', 'Frogger',
    'Frostbite', 'Galaxian', 'Gopher', 'Gravitar', 'Hangman', 'HauntedHouse', 'Hero', 'HumanCannonball',
    'IceHockey', 'Jamesbond', 'JourneyEscape', 'Joust', 'Kaboom', 'Kangaroo', 'KeystoneKapers', 'KingKong',
    'Klax', 'Koolaid', 'Krull', 'KungFuMaster', 'LaserGates', 'LostLuggage', 'MarioBros', 'MazeCraze',
    'MiniatureGolf', 'MontezumaRevenge', 'MrDo', 'MsPacman', 'NameThisGame', 'Othello', 'Pacman', 'Phoenix',
    'Pitfall', 'Pitfall2', 'Pong', 'Pooyan', 'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank',
    'Seaquest', 'SirLancelot', 'Skiing', 'Solaris', 'SpaceInvaders', 'SpaceWar', 'StarGunner', 'Superman',
    'Surround', 'Tennis', 'Tetris', 'TicTacToe3d', 'TimePilot', 'Trondead', 'Turmoil', 'Tutankham', 'UpNDown',
    'Venture', 'VideoCheckers', 'Videochess', 'Videocube', 'VideoPinball', 'Warlords', 'WizardOfWor', 'WordZapper',
    'YarsRevenge', 'Zaxxon'
]

# A popular subset of the Atari ALE
atari_tasks_26 = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]

# Prints all available ROMs
# import glob
# print(sorted([''.join(t.capitalize() for t in task.split('/')[-1].split('.bin')[0].split('_'))
#               for task in glob.glob('../../../Datasets/Suites/Atari_ROMS/*')]))

if __name__ == '__main__':
    out = ""
    for task in atari_tasks:
        f = open(f"./{task.lower()}.yaml", "w")

        write = f"""Env: World.Environments.Atari.Atari
suite_name: atari
task_name: {task}
env:
    game: {task}
discrete: true
action_repeat: 4
truncate_episode_steps: 250
nstep: 10
frame_stack: 3
train_steps: 500000
stddev_schedule: 'linear(1.0,0.1,20000)'
hd_capacity: 1e6

# Atari has two augmentations
Aug: Sequential
aug:
    _targets_: [RandomShiftsAug, IntensityAug]
    pad: 4
    noise: 0.05
"""
        write = f"""imports:
    - atari
env:
    game: {task}"""

        f.write(write)
        f.close()
        out += ' "' + task.lower() + '"'
    print(out)
