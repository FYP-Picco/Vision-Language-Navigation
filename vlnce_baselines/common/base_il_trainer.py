import json
import os
import time
import warnings
import cv2
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import jsonlines
import torch
import torch.nn.functional as F
import tqdm
from gym import Space, spaces
from habitat import Config, logger
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses

from vlnce_baselines.Action_space_creator import ActionSpace
from vlnce_baselines.common.ZED import Cam
import torch
from vlnce_baselines.common.TextTokenz import TextProcessor
from vlnce_baselines.common.video_utils import generate_video_single, append_text_to_images

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


class BaseVLNCETrainer(BaseILTrainer):
    """A base trainer for VLN-CE imitation learning."""

    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.IL.lr
        )
        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.policy.load_state_dict(ckpt_dict["state_dict"])
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.start_epoch = ckpt_dict["epoch"] + 1
                self.step_id = ckpt_dict["step_id"]
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        T, N = corrected_actions.size()

        recurrent_hidden_states = torch.zeros(
            N,
            self.policy.net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        AuxLosses.clear()

        distribution = self.policy.build_distribution(
            observations, recurrent_hidden_states, prev_actions, not_done_masks
        )

        logits = distribution.logits
        logits = logits.view(T, N, -1)

        action_loss = F.cross_entropy(
            logits.permute(0, 2, 1), corrected_actions, reduction="none"
        )
        action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

        aux_mask = (weights > 0).view(-1)
        aux_loss = AuxLosses.reduce(aux_mask)

        loss = action_loss + aux_loss
        loss = loss / loss_accumulation_scalar
        loss.backward()

        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.item()
        return loss.item(), action_loss.item(), aux_loss


    def single_inference(self) -> None:
        """Runs a single inference on a user input and prints a predictions."""

        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.INFERENCE.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")[
                    "config"
                ]
            )
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.INFERENCE.LANGUAGES
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = config.INFERENCE.CKPT_PATH
        config.TASK_CONFIG.TASK.MEASUREMENTS = []
        config.TASK_CONFIG.TASK.SENSORS = [
            s for s in config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s
        ]
        # config.ENV_NAME = "VLNCEInferenceEnv"
        config.freeze()


        observation_space = spaces.Dict({
            'depth': spaces.Box(low = 0.0, high=1.0, shape= (256, 256, 1), dtype=np.float32), 
            'instruction': spaces.Discrete(4), 
            'rgb': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
            })
        

        actionsDict = {
            0:'STOP',
            1:'MOVE_FORWARD',
            2:'TURN_LEFT',
            3:'TURN_RIGHT'
        }
        action_space = ActionSpace(actionsDict)
    

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()

        processor = TextProcessor('data/Vocab_file.txt', torch.device('cuda:0'))
        text = "Go forward and stop near the fire extinguisher."
        #1."Exit the hallway, turn right and walk past the green sofa" 
        #2."Exit the room through the door. Go straight through the hallway and enter the next room. Walk towards the table and stop." #input('Give me an instruction:')#"Exit the room through the door. Go straight through the hallway and enter the next room. Walk towards the table and stop."
        #3.Exit the room through the door. Go straight through the hallway, stop at fire extinguisher.
        depth,rgb = Cam.newFrame() 
        batch = processor.process(text) 
        batch['rgb']=rgb
        batch['depth']=depth   

        #batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        # print(batch)
        num_envs=1
        
        rnn_states = torch.zeros(
            num_envs,
            self.policy.net.num_recurrent_layers,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            num_envs, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            num_envs, 1, dtype=torch.uint8, device=self.device
        )

        rgb_frames = [[]]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
        

        stop = False
        dones = [False]
        steps = 0 
        while not stop:
            #current_episodes = envs.current_episodes()  #episode_id, scene_id, start_pos, start_rotation, instruction with tokens
            with torch.no_grad():
                actions, rnn_states = self.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=not config.INFERENCE.SAMPLE,
                )
                steps+=1
                action_value = actions.item()
                _mapping = {0:'STOP', 1:'MOVE_FORWARD', 2:'TURN_LEFT', 3:'TURN_RIGHT'}
                mapped_act = _mapping.get(action_value,'unknown')      
                print(mapped_act)   
                Cam.closeAllWindows()                     
                #actions = tensor([[3]], device='cuda:0')
                prev_actions.copy_(actions)
              
            # outputs = envs.step([a[0].item() for a in actions]) #take a step in the simulated environment
            # observations, _, dones, infos = [     #observations = camera_input,instruction
            #     list(x) for x in zip(*outputs)
            
            # time.sleep(5)
            if actions[0][0]==0:
                dones = [True]
            stop=dones[0]
            
            if len(config.VIDEO_OPTION) > 0:
                video = []
        
                cpu_tensor_RGB = batch['rgb'].cpu()
                src_rgb = np.float32(cpu_tensor_RGB)[0]
                image_rgb = cv2.cvtColor(src_rgb, cv2.COLOR_BGR2RGB) 
                video.append(image_rgb)
                
                cpu_tensor_depth = batch['depth'].cpu()
                src_depth = np.float32((cpu_tensor_depth)[0].squeeze() * 255).astype(np.uint8)
                src_depth = np.stack([src_depth for _ in range(3)], axis=2)
                depth_map = cv2.resize(src_depth,dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
                video.append(depth_map)
                
                frame = np.concatenate(video,axis=1)
                frame = append_text_to_images(frame, text)
                rgb_frames[0].append(frame)
                
            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=self.device,
            )
                
            depth,rgb = Cam.newFrame()
            batch['rgb']=rgb
            batch['depth']=depth   

            if steps>=3:
                break  
            # batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            # print(batch)
            
        if len(config.VIDEO_OPTION) > 0:
            generate_video_single(
                video_option=config.VIDEO_OPTION,
                video_dir=config.VIDEO_DIR,
                images=rgb_frames[0],
                episode_id=10832,
                checkpoint_idx=45,
            )
    
