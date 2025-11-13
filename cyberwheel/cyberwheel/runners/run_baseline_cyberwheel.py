from cyberwheel.utils import YAMLConfig, parse_default_override_args
from cyberwheel.runners.baseline_runner import BaselineRunner

def run_cyberwheel(args: YAMLConfig):
    # Initialize the Evaluator object
    
    runner = BaselineRunner(args)

    # Configure training parameters and train
    runner.configure()

    runner.run()

    runner.close()
