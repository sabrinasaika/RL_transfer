
def reward_red_delay(rewarder, **kwargs):
    blue_agent_result = kwargs.get("blue_agent_result", None)
    red_agent_result = kwargs.get("red_agent_result", None)

    b = 0
    b_recurring = 0

    if red_agent_result.success and red_agent_result.target_host.decoy:
        b += 100
    if blue_agent_result.success:
        b += rewarder.blue_rewards[blue_agent_result.name][0]
    if blue_agent_result.id == "decoy_limit_exceeded":
        b += -5000
    else:
        pass

    return b, b_recurring