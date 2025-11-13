# adapters/reward_normalizer.py
class RewardNormalizer:
    def from_cbs(self, raw_reward, info=None) -> float:
        try: r = float(raw_reward)
        except: r = 0.0
        return max(-1.0, min(1.0, r))
    def from_cw(self, raw_reward, info=None) -> float:
        try: r = float(raw_reward)
        except: r = 0.0
        return max(-1.0, min(1.0, r))
