# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch profiler settings set on global_profiler.global_tool_config.torch must reach
every role's profiler.tool_config.torch, like the nsys and torch_memory settings already do."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config"

ROLE_TORCH_PATHS = [
    "actor_rollout_ref.actor.profiler.tool_config.torch",
    "actor_rollout_ref.ref.profiler.tool_config.torch",
    "actor_rollout_ref.rollout.profiler.tool_config.torch",
    "critic.profiler.tool_config.torch",
]


def _compose_ppo_trainer(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="ppo_trainer", overrides=overrides)


@pytest.mark.parametrize("torch_path", ROLE_TORCH_PATHS)
def test_global_torch_settings_reach_role_config(torch_path):
    overrides = [
        '++global_profiler.global_tool_config.torch.contents=["cpu","cuda","stack"]',
        "++global_profiler.global_tool_config.torch.discrete=true",
    ]
    cfg = _compose_ppo_trainer(overrides)
    torch_cfg = OmegaConf.select(cfg, torch_path)
    assert list(torch_cfg.contents) == ["cpu", "cuda", "stack"]
    assert torch_cfg.discrete is True


@pytest.mark.parametrize("torch_path", ROLE_TORCH_PATHS)
def test_role_torch_settings_default_when_global_unset(torch_path):
    cfg = _compose_ppo_trainer([])
    torch_cfg = OmegaConf.select(cfg, torch_path)
    assert list(torch_cfg.contents) == []
    assert torch_cfg.discrete is False
