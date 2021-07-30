BASE_WORK_DIR="../v100_log/syn/final1"

# source only
python main.py --config_file config/syn2cityscapes_v100_1/run_task/source_only.yaml --work_dir ${BASE_WORK_DIR}/source_only

# AT warmup
python main.py --config_file config/syn2cityscapes_v100_1/run_task/warmup_at.yaml --resume_from ${BASE_WORK_DIR}/source_only/best_iter.pth --work_dir ${BASE_WORK_DIR}/warmup_at

# self-training round 1 
python main.py --config_file config/syn2cityscapes_v100_1/run_task/sl_1.yaml --resume_from ${BASE_WORK_DIR}/warmup_at/last_iter.pth --pseudo_resume_from ${BASE_WORK_DIR}/warmup_at/best_iter.pth --work_dir ${BASE_WORK_DIR}/sl_1

# self-training round 2 
python main.py --config_file config/syn2cityscapes_v100_1/run_task/sl_2.yaml --resume_from ${BASE_WORK_DIR}/sl_1/epoch_2.pth --pseudo_resume_from ${BASE_WORK_DIR}/sl_1/epoch_1.pth --work_dir ${BASE_WORK_DIR}/sl_2

# self-training round 3
python main.py --config_file config/syn2cityscapes_v100_1/run_task/sl_3.yaml --resume_from ${BASE_WORK_DIR}/sl_2/epoch_2.pth --pseudo_resume_from ${BASE_WORK_DIR}/sl_2/epoch_1.pth --work_dir ${BASE_WORK_DIR}/sl_3
