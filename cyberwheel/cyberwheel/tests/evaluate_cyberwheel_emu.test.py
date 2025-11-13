# CleanRL script for training Cage Challenge 2 agents. CleanRL documentation can be found at https://docs.cleanrl.dev/,
import os
import time

from cyberwheel.utils import (
    parse_override_args,
    parse_eval_override_args,
    YAMLConfig,
)
from cyberwheel.runners import RLEvaluator

"""
This script will train cyberwheel. Using the args from the config file passed, it will run an Actor-Critic RL algorithm and run training
with intermittent evaluations/saves. If tracking to W&B, this will be logged in your W&B project for each training run.
"""
# Allows using command line to override args in the YAML config
override_args = parse_eval_override_args()
# args = YAMLConfig(override_args.eval_config)
args = YAMLConfig("evaluate_blue.yaml")
args.parse_config()
args_dict = vars(args)
override_args_dict = vars(override_args)
for arg in override_args_dict:
    if arg in args_dict and override_args_dict[arg]:
        setattr(args, arg, override_args_dict[arg])

args.evaluation = True  # Should be set anyway?

# Initialize the Evaluator object
evaluator = RLEvaluator(args)

# Configure training parameters and train
evaluator.configure_evaluation()

# evaluator.evaluate()
