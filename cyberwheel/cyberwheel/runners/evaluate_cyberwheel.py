from cyberwheel.utils import YAMLConfig #, Evaluator, EmulatorEvaluator, parse
from cyberwheel.runners.rl_evaluator import RLEvaluator
import signal
import sys


def evaluate_cyberwheel(args: YAMLConfig, emulate=False):
    """
    This script will evaluate cyberwheel. Using the args from the config file passed, it will evaluate a pre-trained model and evaluate.
    Can fetch models from W&B, as well as use any stored in cyberwheel/data/models
    """
    
    #print('Press Ctrl+C')
    #signal.pause()
    args.evaluation = True



    # Initialize the Evaluator object
    evaluator = RLEvaluator(args) #EmulatorEvaluator(args) if emulate else Evaluator(args)

    # Configure training parameters and train
    
    evaluator.configure_evaluation()

    def signal_handler(sig, frame):
        print('Closing gracefully...')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    evaluator.load_models()
    evaluator._initialize_environment()

    evaluator.evaluate()
