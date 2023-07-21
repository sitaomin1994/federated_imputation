from src.experiment_management.result_management import process_ret_to_xlsx

if __name__ == '__main__':
	experiment_name = "ms_eval_partition1"
	experiment_marker = ""
	dir = experiment_name + "/" + experiment_marker
	level_dirs = ['0306', 'segmentation', 'mary', 'logistic']
	process_ret_to_xlsx(dir, level_dirs)
