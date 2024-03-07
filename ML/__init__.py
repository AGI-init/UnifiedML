# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""
UnifiedML - A unified framework for intelligence training. A built and conceived generalist agent by Sam Lerman.
    This file makes it possible to import UnifiedML as a package or launch it within Python.
"""
import sys
import os
import multiprocessing as mp

from minihydra import len_return_variables, reset_portal

sys.path.append(os.path.dirname(__file__))

_, dirs, files = next(os.walk(os.path.dirname(__file__)))

for file in files:
    if file != '__init__.py' and file[-3:] == '.py':
        globals().update({file[:-3]: __import__(file[:-3])})

for dir in dirs:
    globals().update({dir: __import__(dir)})


# Executes from code
def ML(args=None, **kwargs):
    # Only run in __main__ call and MainProcess, not imports or forks
    _main_ = sys._getframe(1).f_globals["__name__"] == '__main__' and mp.current_process().name == 'MainProcess'
    from Run import main
    if _main_:
        outs = main(args, **kwargs)
        reset_portal()
        return outs
    # If not running, still try to return the correct number of outputs; assumes consistent return sizes
    return (None,) * len_return_variables(main)


run = launch = main = ml = ML  # Pseudonyms

from Agents import Agent
from Utils import load, save, optimize, normalize, standardize
from Benchmarking.Plot import plot as Plot
from Benchmarking.Plot import plot
