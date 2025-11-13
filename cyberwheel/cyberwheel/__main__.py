import sys
from cyberwheel.utils import parse_default_override_args, parse_eval_override_args, parse_override_args, parse
from cyberwheel.runners import train_cyberwheel, evaluate_cyberwheel, run_cyberwheel, run_visualization_server
def display_help():
    sys.argv = ['']
    print("---------------------------------------------------------------------------------------------------\nTraining Cyberwheel:\n\n")
    parse_override_args(print_help=True)
    print("---------------------------------------------------------------------------------------------------\nEvaluating Cyberwheel:\n\n")
    parse_eval_override_args(print_help=True)
    print("---------------------------------------------------------------------------------------------------\nRunning Cyberwheel:\n\n")
    parse_default_override_args(print_help=True)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        mode = sys.argv.pop(1)
        config = sys.argv.pop(1)
        if mode == 'visualizer':
            run_visualization_server(config)
            sys.exit(0)

        args = parse(config, mode) if mode in ['train', 'evaluate', 'emulate', 'run'] else None

        if mode == 'train':
            train_cyberwheel(args)
        elif mode == 'evaluate':
            evaluate_cyberwheel(args, emulate=False)
        elif mode == 'emulate':
            evaluate_cyberwheel(args, emulate=True)
        elif mode == 'run':
            run_cyberwheel(args)
        else:
            display_help()