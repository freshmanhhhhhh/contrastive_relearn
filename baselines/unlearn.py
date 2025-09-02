import hydra
from src import it_unlearn


@hydra.main(version_base=None, config_path="config", config_name="forget_lora")
def main(cfg):
    # print(cfg)
    it_unlearn(cfg)

if __name__ == "__main__":
    main()
