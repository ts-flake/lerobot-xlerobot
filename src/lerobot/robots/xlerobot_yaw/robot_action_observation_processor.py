from dataclasses import dataclass, field
from typing import Any
from pprint import pformat
import logging

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    EnvTransition,
    RobotObservation,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)

@ProcessorStepRegistry.register("robot_action_feature_select")
@dataclass
class RobotActionFeatureSelect(RobotActionProcessorStep):
    features_to_ignore: list

    def action(self, action: RobotAction) -> RobotAction:
        for feat in self.features_to_ignore:
            action.pop(feat, None)
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in self.features_to_ignore:
            features[PipelineFeatureType.ACTION].pop(feat, None)
        return features
    
@ProcessorStepRegistry.register("robot_observation_feature_select")
@dataclass
class RobotObservationFeatureSelect(ObservationProcessorStep):
    features_to_ignore: list

    def observation(self, observation: RobotObservation) -> RobotObservation:
        for feat in self.features_to_ignore:
            observation.pop(feat, None)
        return observation
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in self.features_to_ignore:
            features[PipelineFeatureType.OBSERVATION].pop(feat, None)
        return features

