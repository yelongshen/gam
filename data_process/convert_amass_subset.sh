#!/bin/bash

sed -i 's/joint_pos = motion_data\["joint_pos"\].*/joint_pos = list(motion_data.values())[0]["joint_pos"] if isinstance(motion_data, dict) else tuple(motion_data)[1]["joint_pos"]/g' gear_sonic_deploy/reference/convert_motions.py
mkdir -p gear_sonic_deploy/reference/evaluation_set/Basic_Locomotion
echo 'Converting Basic_Locomotion...' 
.venv_sim/bin/python gear_sonic_deploy/reference/convert_motions.py /home/grease/GR00T-WholeBodyControl/data/smpl_filtered/crouch_ff_start_270_R_001__A198.pkl gear_sonic_deploy/reference/evaluation_set/Basic_Locomotion
.venv_sim/bin/python gear_sonic_deploy/reference/convert_motions.py /home/grease/GR00T-WholeBodyControl/data/smpl_filtered/crouch_ff_stop_225_001__A146.pkl gear_sonic_deploy/reference/evaluation_set/Basic_Locomotion
mkdir -p gear_sonic_deploy/reference/evaluation_set/Agility_HighDynamic
echo 'Converting Agility_HighDynamic...' 
.venv_sim/bin/python gear_sonic_deploy/reference/convert_motions.py /home/grease/GR00T-WholeBodyControl/data/smpl_filtered/dance_hiphop_kick_it_variation_R_001__A312.pkl gear_sonic_deploy/reference/evaluation_set/Agility_HighDynamic
.venv_sim/bin/python gear_sonic_deploy/reference/convert_motions.py /home/grease/GR00T-WholeBodyControl/data/smpl_filtered/inj_torso_jump_ff_180_R_max_002__A075_M.pkl gear_sonic_deploy/reference/evaluation_set/Agility_HighDynamic
mkdir -p gear_sonic_deploy/reference/evaluation_set/UpperBody_Manipulation
echo 'Converting UpperBody_Manipulation...' 
.venv_sim/bin/python gear_sonic_deploy/reference/convert_motions.py /home/grease/GR00T-WholeBodyControl/data/smpl_filtered/reach_jump_R_003__A160.pkl gear_sonic_deploy/reference/evaluation_set/UpperBody_Manipulation
.venv_sim/bin/python gear_sonic_deploy/reference/convert_motions.py /home/grease/GR00T-WholeBodyControl/data/smpl_filtered/reach_jump_R_001__A046_M.pkl gear_sonic_deploy/reference/evaluation_set/UpperBody_Manipulation
