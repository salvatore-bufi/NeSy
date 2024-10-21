from elliot.run import run_experiment

config_file = 'bprmf_fb'
# config_file = 'rbrs_fb'
run_experiment(f"config_files/" + config_file + ".yml")
