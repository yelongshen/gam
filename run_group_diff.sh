#!/usr/bin/env bash
OLD="/home/grease/g1_robot_data/g1_run_0917/g1_deploy_run_09182026_081125,/home/grease/g1_robot_data/g1_run_0917/g1_deploy_run_09182026_092156,/home/grease/g1_robot_data/g1_run_0917/g1_deploy_run_09182026_092516"
NEW="/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_074107,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_074421,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_074919,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_080352,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_081337,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_081856,/home/grease/humanoid-foundation-model-2/online_robodata/g1_run_0918/20260919_082250"
cd /home/grease/gam
.venv_sim/bin/python model_eval/sim2real_frame0_group_diff.py "$OLD" "$NEW"
