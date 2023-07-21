import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="exp_config_imp_eval")
def my_app(cfg: DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
	my_app()
