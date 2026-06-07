# Copyright 2026 - Two-Model Alternate Training Recipe
"""
Entry point for two-model alternate training.

Reuses verl's `main_ppo.TaskRunner` for setting up workers / datasets /
tokenizer, then swaps in the `TwoModelAlternateTrainer` instead of the
standard `RayPPOTrainer`.
"""

from __future__ import annotations

import os
import socket
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Make the local `two_model` package importable when this file is run as
# a module from the project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from two_model_trainer import TwoModelAlternateTrainer


class TwoModelTaskRunner(TaskRunner):
    """Subclass of verl's TaskRunner that uses our two-model trainer."""

    def run(self, config):
        from pprint import pprint

        from verl.utils.fs import copy_to_local
        from verl.trainer.ppo.utils import need_critic, need_reference_policy

        print(f"TwoModelTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        if need_critic(config):
            self.add_critic_worker(config)

        self.add_reward_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        thinker_path = config.two_model.thinker_model_path
        solver_path = config.two_model.solver_model_path
        thinker_local_path = copy_to_local(thinker_path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        solver_local_path = copy_to_local(solver_path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        role_tokenizers = {
            "thinker": hf_tokenizer(thinker_local_path, trust_remote_code=trust_remote_code),
            "solver": hf_tokenizer(solver_local_path, trust_remote_code=trust_remote_code),
        }
        role_processors = {
            "thinker": hf_processor(thinker_local_path, trust_remote_code=trust_remote_code, use_fast=True),
            "solver": hf_processor(solver_local_path, trust_remote_code=trust_remote_code, use_fast=True),
        }
        # Dataset filtering only sees the original problem before the
        # two-stage recipe expands it, so use the Thinker tokenizer here.
        tokenizer = role_tokenizers["thinker"]
        processor = role_processors["thinker"]

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = TwoModelAlternateTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            role_tokenizers=role_tokenizers,
            role_processors=role_processors,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path=None, config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_two_model(config)


def run_two_model(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        runtime_env.setdefault("env_vars", {})
        runtime_env.env_vars.update(
            {
                "TOKENIZERS_PARALLELISM": "false",
                "RAYON_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = ray.remote(num_cpus=1)(TwoModelTaskRunner).remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    # Provide hydra a config_path that points at the standard verl PPO
    # config so we can reuse all defaults.  We do this manually so the
    # user does not need to copy a yaml.
    main()
