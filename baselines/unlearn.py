import hydra
from src import it_unlearn


@hydra.main(version_base=None, config_path="config", config_name="forget_lora")
def main(cfg):
    print(type(cfg))
    print(cfg)
    
    it_unlearn(cfg)

if __name__ == "__main__":
    main()


"""
{'model_family': 'kud-llama2-7b', 'model_path': '../../paper_models/kud-llama2-7b_lora_privacy/', 'LoRA': {'r': 32, 'alpha': 32, 'dropout': 0.05}, 'lr': 1e-05, 'forget_data_path': '../../dataset/augument_data/contrastive_data/KnowUnDo_privacy.jsonl', 'retain_data_path': '../../dataset/KnowUnDo/privacy/retention_train.json', 'idonknow_file_path': '../../dataset/idontknow.txt', 'batch_size': 1, 'num_epochs': 4, 'gradient_accumulation_steps': 4, 'loss_type': 'contrastive', 'save_dir': '../../memory/kud-llama2-7b_contrastive_privacy_512_1e-5', 'weight_decay': 0.01, 'save_model': True, 'eval_while_train': False, 'eval_only': False, 'override': True, 'overwrite_dir': True, 'max_length': 512, 'seed': 42, 'ds_config': '../config/ds_z0_config.json', 'resume_from_checkpoint': None}
"""