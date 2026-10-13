#!/usr/bin/env bash
cd /home/grease/gam
OLD="/home/grease/g1_robot_data/g1_run_0908/g1_deploy_run_09082026_run1,/home/grease/g1_robot_data/g1_run_0908/g1_deploy_run_09082026_run2,/home/grease/g1_robot_data/g1_run_0908/g1_deploy_run_09082026_run3,/home/grease/g1_robot_data/g1_run_0908/g1_deploy_run_09082026_run4,/home/grease/g1_robot_data/g1_run_0911/g1_deploy_run_09112026_run1,/home/grease/g1_robot_data/g1_run_0911/g1_deploy_run_09112026_run2,/home/grease/g1_robot_data/g1_run_0911/g1_deploy_run_09112026_run3,/home/grease/g1_robot_data/g1_run_0911/g1_deploy_run_09112026_run4"
NEW="/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_074107,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_074421,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_074919,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_080352,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_081337,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_081856,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_082250"
echo "=== FRAME 0 ==="
.venv_sim/bin/python model_eval/sim2real_frame0_group_diff.py "$OLD" "$NEW"
echo
echo "=== STEADY-STATE (encoder_mode==0 mean) ==="
.venv_sim/bin/python model_eval/sim2real_frame0_group_diff.py "$OLD" "$NEW" --mode0mean
