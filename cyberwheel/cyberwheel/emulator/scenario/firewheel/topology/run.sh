#!/bin/bash

# This script starts the CyberWheel experiement in FireWheel.
# The configuration file must be included when executing the script:
#
#                ./run <absolute_location/config_name.yaml>
#
# The available configurations can be found in the /cyberwheel/data/configs/network folder.

NETWORK_CONFIG=$1

if [[ $# -eq 0 ]]; then
    echo "Missing network config name argrument. Include the absolute location along with config name. 
See 'cyberwheel/data/configs/network' folder for available configs and use './run <absolute_location/config_name>.yaml'.
(e.g. ./run path_to_cyberwheel/cyberwheel/data/configs/network/emulator_example_config.yaml)"
    exit 1
fi

if [[ $# -gt 1 ]]; then
    echo "Invalid number of agruments, there is only 1 argument for the network config file.
Include the absolute location along with config name. See 'cyberwheel/data/configs/network' folder and use './run <absolute_location/config_name>.yaml'.
(e.g. ./run path_to_cyberwheel/cyberwheel/data/configs/network/emulator_example_config.yaml)"
    exit 1
fi

if [ ! -f "$NETWORK_CONFIG" ]; then
    echo "Network config file not found. Ensure file exist and that '.yaml' is included in the name.
(e.g. ./run path_to_cyberwheel/data/configs/network/emulator_example_config.yaml)"
    exit 1
fi

export NETWORK_CONFIG=$1
echo "exported NETWORK_CONFIG=$NETWORK_CONFIG"

echo "Launching firewheel experiment..."
firewheel experiment -f cyberwheel.topology control_network minimega.launch
