BASE_WORK_DIR="../t4_log/gta5/final1"

# self-training round 1 
python main.py --config_file config/gtav2cityscapes_t4/run_task/sl_1.yaml --resume_from pretrained_models/M_gtav_at_warmup_t4.pth --pseudo_resume_from pretrained_models/G_gtav_at_warmup_t4.pth --work_dir ${BASE_WORK_DIR}/sl_1

# self-training round 2 
python main.py --config_file config/gtav2cityscapes_t4/run_task/sl_2.yaml --resume_from ${BASE_WORK_DIR}/sl_1/epoch_2.pth --pseudo_resume_from ${BASE_WORK_DIR}/sl_1/epoch_1.pth --work_dir ${BASE_WORK_DIR}/sl_2

# self-training round 3
python main.py --config_file config/gtav2cityscapes_t4/run_task/sl_3.yaml --resume_from ${BASE_WORK_DIR}/sl_2/epoch_2.pth --pseudo_resume_from ${BASE_WORK_DIR}/sl_2/epoch_1.pth --work_dir ${BASE_WORK_DIR}/sl_3
