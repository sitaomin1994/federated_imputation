from src.missing_data_exploration.plotting import make_plots

if __name__ == '__main__':
    plot_params = {
        "exp_dir": "ms_eval_exp_new/0316/breast",
        "levels_dir": "",
        "plot_params": {
            "type": "line_imp_vs_ms_ratio",
            "field": "imputation",
            "filter": {
                "ms_mechanism": ["mary"],
                "imputation": ["iterative-sklearn", "iterative-sklearn-ds", "simple"],
                "classifier": ["logistic"],
                "dataset_name": [],
            }
        }
    }

    make_plots.make_plots_factory(**plot_params)
