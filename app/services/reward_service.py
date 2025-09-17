from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from app.utils.manifest import Manifest


@dataclass
class RewardCalculator:
    manifest: Manifest

    def calculate(self, observations: Dict[str, float] | Dict[str, Dict[str, float]]) -> Dict[str, float]:
        reward_meta = self.manifest.environment.reward_function
        name = reward_meta.name or "RewardFunction"

        if isinstance(next(iter(observations.values())), dict):
            obs_map = {k: v for k, v in observations.items()}
        else:
            obs_map = {"0": observations}  # wrap single-agent data

        if name == "V2GPenaltyReward":
            calculator = V2GPenaltyReward(**reward_meta.params)
            obs_list = [obs_map[str(idx)] for idx in range(len(obs_map))]
            rewards = calculator.calculate(obs_list)
            return {str(idx): float(value) for idx, value in enumerate(rewards)}

        if name == "RewardFunction":
            return {
                idx: -float(obs.get("net_electricity_consumption", 0.0))
                for idx, obs in obs_map.items()
            }

        raise NotImplementedError(f"Reward function '{name}' not supported in inference service.")


class V2GPenaltyReward:
    """Ported reward logic compatible with inference-time calculations."""

    def __init__(
        self,
        peak_percentage_threshold: float = 0.10,
        ramping_percentage_threshold: float = 0.10,
        peak_penalty_weight: float = 20,
        ramping_penalty_weight: float = 15,
        energy_transfer_bonus: float = 10,
        window_size: int = 6,
        penalty_no_car_charging: float = -5,
        penalty_battery_limits: float = -2,
        penalty_soc_under_5_10: float = -5,
        reward_close_soc: float = 10,
        reward_self_ev_consumption: float = 5,
        community_weight: float = 0.2,
        reward_extra_self_production: float = 5,
        squash: int = 0,
    ):
        self.PEAK_PERCENTAGE_THRESHOLD = peak_percentage_threshold
        self.RAMPING_PERCENTAGE_THRESHOLD = ramping_percentage_threshold
        self.PEAK_PENALTY_WEIGHT = peak_penalty_weight
        self.RAMPING_PENALTY_WEIGHT = ramping_penalty_weight
        self.ENERGY_TRANSFER_BONUS = energy_transfer_bonus
        self.WINDOW_SIZE = window_size
        self.PENALTY_NO_CAR_CHARGING = penalty_no_car_charging
        self.PENALTY_BATTERY_LIMITS = penalty_battery_limits
        self.PENALTY_SOC_UNDER_5_10 = penalty_soc_under_5_10
        self.REWARD_CLOSE_SOC = reward_close_soc
        self.REWARD_SELF_EV_CONSUMPTION = reward_self_ev_consumption
        self.COMMUNITY_WEIGHT = community_weight
        self.REWARD_EXTRA_SELF_PRODUCTION = reward_extra_self_production
        self.SQUASH = squash
        self.rolling_window: List[float] = []

    def calculate(self, observations: List[Dict[str, float]]) -> List[float]:
        raw_reward_list = [self._building_reward(obs) for obs in observations]
        reward_list = self._community_reward(observations, raw_reward_list)
        if self.SQUASH:
            import numpy as np

            reward_list = [float(np.tanh(r)) for r in reward_list]
        return reward_list

    def _building_reward(self, observation: Dict[str, float]) -> float:
        net_energy = observation.get("net_electricity_consumption", 0.0)
        reward_type = observation.get("reward_type")
        reward = 0.0

        if reward_type == "C":
            price = observation.get("electricity_pricing", 0.0)
            reward = -price * net_energy if net_energy > 0 else 0.80 * price * abs(net_energy)
        elif reward_type == "G":
            carbon_intensity = observation.get("carbon_intensity", 0.0)
            reward = carbon_intensity * (net_energy * -1)
        elif reward_type == "Z":
            reward = -net_energy if net_energy > 0 else abs(net_energy) * 0.5
        else:
            reward = net_energy * -1

        reward += self._ev_penalty(observation, reward, net_energy)
        return reward

    def _ev_penalty(self, observation: Dict[str, float], current_reward: float, net_energy: float) -> float:
        penalty = 0.0
        penalty_multiplier = abs(current_reward)

        chargers = observation.get("chargers", []) or []
        for charger in chargers:
            last_connected_car = charger.get("last_connected_car")
            last_charged_value = charger.get("last_charging_action_value", 0.0)

            if last_connected_car is None and last_charged_value != 0:
                penalty += self.PENALTY_NO_CAR_CHARGING * penalty_multiplier

            if last_connected_car is not None:
                soc = last_connected_car.get("soc", 0.0)
                capacity = last_connected_car.get("capacity", 1.0)
                required_soc = last_connected_car.get("required_soc_departure", 0.0)

                if soc + last_charged_value > capacity or soc + last_charged_value < 0:
                    penalty += self.PENALTY_BATTERY_LIMITS * penalty_multiplier

                soc_diff = (soc / capacity) - required_soc if capacity else 0.0
                penalty += self.REWARD_CLOSE_SOC * (1 - abs(soc_diff))

        return penalty

    def _community_reward(self, observations: List[Dict[str, float]], rewards: List[float]) -> List[float]:
        community_net_energy = sum(obs.get("net_electricity_consumption", 0.0) for obs in observations)

        if len(self.rolling_window) >= self.WINDOW_SIZE:
            self.rolling_window.pop(0)
        self.rolling_window.append(community_net_energy)

        average_past_consumption = sum(self.rolling_window) / len(self.rolling_window)
        dynamic_peak_threshold = average_past_consumption * (1 + self.PEAK_PERCENTAGE_THRESHOLD)
        ramping = community_net_energy - average_past_consumption

        community_reward = 0.0
        if community_net_energy > dynamic_peak_threshold:
            community_reward -= (community_net_energy - dynamic_peak_threshold) * self.PEAK_PENALTY_WEIGHT
        if abs(ramping) > dynamic_peak_threshold:
            community_reward -= abs(ramping) * self.RAMPING_PENALTY_WEIGHT

        community_reward += sum(
            -obs.get("net_electricity_consumption", 0.0) * self.ENERGY_TRANSFER_BONUS
            for obs in observations
            if obs.get("net_electricity_consumption", 0.0) < 0
        )

        return [r + community_reward * self.COMMUNITY_WEIGHT for r in rewards]
