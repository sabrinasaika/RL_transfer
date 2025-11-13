from cyberwheel.network.network_base import Host

def reward_decoy_hits(rewarder, **kwargs):
    red_agent_result = kwargs.get("red_agent_result", None)
    valid_targets = kwargs.get("valid_targets", [])
    
    red_action = red_agent_result.action.get_name()
    red_valid_target = isinstance(red_agent_result.target_host, Host)

    if red_valid_target:
        red_target = red_agent_result.target_host.name
        red_target_is_decoy = red_agent_result.target_host.decoy
    else:
        red_target = "invalid"
        red_target_is_decoy = False
    
    # Safely look up reward mapping; default to (0, 0) if action key not present
    immediate_recurring = rewarder.red_rewards.get(red_action, (0, 0))
    if red_agent_result.success and red_target in valid_targets: # If red action succeeded on a Host
        r = immediate_recurring[0] * 1
        r_recurring = immediate_recurring[1] * 1
    elif red_agent_result.success and red_target not in valid_targets:
        r = immediate_recurring[0] * 0
        r_recurring = immediate_recurring[1] * 0
    else: # Red is not successful
        # Softer failure penalty to avoid overwhelming sparse successes
        r = -0.1
        r_recurring = 0

    # TODO: If 'prioritize_early_success' flag is set, else red_multiplier is 1

    #rewarder.red_agent.observation.update_obs(current_step=rewarder.current_step, total_steps=rewarder.red_agent.args.num_steps)

    # Remove early-phase multiplier to prevent excessive negatives early
    # (keeps reward scale stable across episode)
    # red_multiplier = 1
    # r *= red_multiplier

    return r, r_recurring
    