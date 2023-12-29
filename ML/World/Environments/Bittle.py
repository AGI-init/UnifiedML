# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import asyncio
import threading
import time

from bleak import BleakScanner, BleakClient

import numpy as np

from minihydra import Args


class Bittle:
    """
    A general-purpose environment:

    Must accept: **kwargs as init arg.

    Must have:

    (1) a "step" function, action -> exp
    (2) "reset" function, -> exp
    (3) "render" function, -> image
    (4) "episode_done" attribute
    (5) "obs_spec" attribute which includes:
        - "shape", "mean", "stddev", "low", "high" (the last 4 can be None)
    (6) "action-spec" attribute which includes:
        - "shape", "discrete_bins" (should be None if not discrete), "low", "high", and "discrete"
    (7) "exp" attribute containing the latest exp

    Recommended: Discrete environments should have a conversion strategy for adapting continuous actions (e.g. argmax)

    An "exp" (experience) is an Args consisting of "obs", "action" (prior to adapting), "reward", and "label"
    as numpy arrays with batch dim or None. "reward" is an exception: should be numpy array, can be empty/scalar/batch.

    ---

    Can optionally include a frame_stack, action_repeat method.

    """
    def __init__(self, **kwargs):
        self.episode_done = False

        print('Discovering devices...')

        self.bittle = discover_devices()
        self.bluetooth, self.bluetooth_thread = connect(self.bittle.address)

        print('Connected to Bittle!')

        time.sleep(10)  # Give Bittle time to boot up

        self.reader, self.writer = [service.characteristics for service in self.bluetooth.services][0]

        parallelize(self.bluetooth.start_notify(self.reader, self.reading))  # Asynchronous reading Bittle measurements

        self.measured = True
        self.action_done = True
        self.action_start_time = time.time()

        self.obs_spec = {'shape': (3,),  # TODO (6,), currently just gyroscope. (15,) with action!
                         'mean': None,
                         'stddev': None,
                         'low': None,
                         'high': None}

        self.action_spec = {'shape': (9,),
                            'discrete_bins': None,
                            # 'low': -25,
                            # 'high': 25,
                            'low': -180,
                            'high': 180,
                            'discrete': False}

        self.exp = Args()  # Experience dictionary

        self.reward = 0

        # self.frames = deque([], frame_stack or 1)  # TODO

        # Maybe add atexit

    def reading(self, _, data: bytearray):
        measurement = data.decode('ISO-8859-1')

        if 'i' in measurement or 'Ready' in measurement:
            self.action_done = True
        elif all(char.isdigit() or char in '-.v\r\n\t' for char in measurement):
            obs = getattr(self.exp, 'obs', None)

            if obs is None or isinstance(obs, np.ndarray):
                self.exp.obs = measurement
            else:
                self.exp.obs += measurement  # Bittle sometimes sends IMU measurements in multiple increments

            if 'v' in measurement:
                self.exp.obs = np.array(list(map(float, self.exp.obs.strip('v\r\n').split('\t'))))
                self.measured = True

    def step(self, action=None):
        self.action_start_time = time.time()

        if action is None:
            # Random action
            action = np.random.rand(*self.action_spec['shape']) * (self.action_spec['high'] - self.action_spec['low']) \
                     + self.action_spec['low']

        # Constrain action ranges to prevent collisions
        constrain(action)

        self.action_done = False
        asyncio.run(self.bluetooth.write_gatt_char(self.writer, encode(action)))  # Triggers a reaction

        self.exp.action = action
        self.exp.label = None

        while not self.action_done and time.time() - self.action_start_time < 1:  # Action shouldn't take more than 1s
            pass

        self.measured = False
        asyncio.run(self.bluetooth.write_gatt_char(self.writer, b'v'))

        # Keep trying to measure if fail
        while not self.measured:
            if time.time() - self.action_start_time > 1:  # Measure shouldn't take more than 1s
                self.action_start_time = time.time()
                asyncio.run(self.bluetooth.write_gatt_char(self.writer, b'v'))

        # self.reward = max(self.exp.obs[0], self.reward)
        # self.exp.reward = int(self.exp.obs[0] >= self.reward)  # Positive forward velocity?

        return self.exp  # Experience

    def reset(self):
        ...

    def render(self):
        ...

    def disconnect(self):
        while not self.action_done:
            pass

        parallelize(self.bluetooth.disconnect())
        self.bluetooth_thread.set()

        print('Disconnected from Bittle')


servos = range(16)  # 16 degrees of freedom


def encode(rotations: list):
    rotations = np.insert(rotations, 1, [0] * 7)
    return ('i ' + ' '.join(map(str, [*sum(zip(servos, np.round(rotations)), ())]))).encode('utf-8')


# Discover Petoi Bittle device
def discover_devices():
    async def _discover_devices():
        devices = await BleakScanner.discover(timeout=15)
        for d in devices:
            if d.name and 'Petoi' in d.name:
                return d
        assert False, 'Could not find Petoi device in *disconnected* Bluetooth devices. ' \
                      'Make sure Your Bittle is on and your machine is *not* yet connected to it. ' \
                      'Can only auto-connect disconnected devices. Consider restarting the Bittle or your Bluetooth.'
    return asyncio.run(_discover_devices())


# Connect to a Bluetooth address
def connect(address):
    bluetooth = BleakClient(address)

    async def _connect():
        await bluetooth.connect()

    event = parallelize(_connect(), forever=True)

    return bluetooth, event


# Helper function to launch coroutines in a thread non-disruptively (since Bleak uses coroutines)
def parallelize(run, forever=False):
    event = asyncio.Event()
    event.clear()

    async def launch():
        try:
            await run
        except RuntimeError as e:
            if 'got Future <Future pending> attached to a different loop' not in str(e):
                raise e
        if forever:
            await event.wait()  # Keep running

    thread = threading.Thread(target=asyncio.run, args=(launch(),))
    thread.start()

    return event


"""Action constraints
Goals:
(a) Avoid collisions between front legs and back legs
(b) Avoid collisions between legs and ankles with body
(c) Still allow robust movements such as needed for jumping, flipping, etc. 
(d} Prevent wire strain"""


# body = ['head',
#         'nothing', 'nothing', 'nothing',
#         'nothing', 'nothing', 'nothing', 'nothing',
#         'front left leg', 'front right leg', 'back right leg', 'back left leg',
#         'front left ankle', 'front right ankle', 'back right ankle', 'back left ankle']

# Back leg: [-90, 180] "straight forward", "vertically back"
# Back ankle: [-45, 180] "bent slightly upwards", "bent backwards"

# Front leg: [-180, 65] "vertically up", "bent back"
# Front ankle: [-45, 180] "bent slightly upwards", "bent backwards"

# ---------

# Limit: Back: ankle > -90 - 2 * leg  To avoid ankle underside collision
# Limit: Back: leg, ankle... Due to wires, some stretches are physically impossible. TODO
# Limit: If Back leg < 0 and Front leg > 0, Back leg < Front leg - 80, Front ankle < 10  TODO

head = [0]
front_legs = [1, 2]  # left, right
back_legs = [3, 4]  # right, left
front_ankles = [5, 6]  # left, right
back_ankles = [7, 8]  # right, left

head, front_legs, back_legs, front_ankles, back_ankles = map(np.array, (head, front_legs, back_legs, front_ankles,
                                                                        back_ankles))

ranges = [(head, [-25, 25]), (front_legs, [-180, 65]), (back_legs, [-90, 180]), (front_ankles, [-45, 180]),
          (back_ankles, [-45, 180])]

constraints = [
    # Avoid back ankle underside collision
    [tuple(), ((back_ankles, lambda a: np.maximum(a[back_ankles], -90 - 2 * a[back_legs])),)],
    #
    # [((back_legs, lambda a: a < 0), (front_legs, lambda a: a > 0)),
    #  ((back_legs, lambda a: np.minimum(a[back_legs], a[front_legs] - 80)),
    #   (front_ankles, lambda a: np.minimum(a[front_ankles], 10)))]
]

# TODO RE-CALIBRATE!


# Constrains actions to reasonable ranges to avoid collisions
def constrain(action):
    for joint, (low, high) in ranges:
        # action[joint] = (action[joint] + 180) * ((high - low) / 360) + low  # Changes scale of constraints...
        action[joint] = np.minimum(action[joint], high)
        action[joint] = np.maximum(action[joint], low)

    for constraint in constraints:
        conditions, norms = constraint

        truth = np.array([1, 1], dtype=bool)
        for part, condition in conditions:
            truth = truth & condition(action[part])

        if not truth.any():
            continue

        for part, norm in norms:
            action[part[truth]] = norm(action)


# Useful, scales from zero-center TODO
# def zero_scale(action):
#     for joint, (low, high) in ranges:
#         action[joint] -= (high - low) / 2


if __name__ == '__main__':
    bittle = Bittle()
    while True:
        # Random action
        bittle.step()

        # try:
        #     command = np.array(list(map(int, input('enter 16-digit command: ').strip('[]').split(', '))), 'float32')
        # except ValueError:
        #     continue
        # # zero_scale(command)
        # bittle.step(command)

        # Can launch custom commands

        # commands = [np.array(command, dtype='float32') for command in [[0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                                                ]]

        # for command in commands:
        #     bittle.step(command)
    # bittle.disconnect()  # Daemon threads https://stackoverflow.com/a/2564282/22002059
