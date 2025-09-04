import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer
from src.utils import get_batch_loss
import copy
import deepspeed

class IterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 **kwargs):
        self.loss_type = kwargs.pop("loss_type", "ga")
        self.ref_model = kwargs.pop("ref_model", None)
        self.beta = kwargs.pop("beta", 0.1)    # Only relevant when `'po' in self.loss_type`

        super().__init__(*args, **kwargs)
        if self.ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            # ref_model = ref_model.eval()
            self.ref_model = self.e_prepare_deepspeed(self.ref_model)



    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model

    def compute_loss(self, model, x, return_outputs=False, num_items_in_batch=None):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        ### 1. Split the input ###
        
        if self.loss_type in ["dpo","dpo_gdr","dpo_klr"]:
            x_f, x_r, x_i = x # 遗忘数据, 保留数据, "我不知道"数据
        elif self.loss_type in ["relearn_dpo", "relearn_dpo_gdr", "relearn_dpo_klr"]:
            x_p, x_n, x_r = x # 偏好数据, 负偏好数据, 保留数据
        else:
            x_f, x_r = x # 遗忘数据, 保留数据

        ### 2. Calculate Loss Based on Loss Type ###
        ### ga,ga_klr,ga_gdr
        if self.loss_type == 'ga':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss_f = outputs_f.loss # Loss_gdf
            loss = -loss_f # 梯度上升

        elif self.loss_type == 'ga_gdr':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss_f = outputs_f.loss # Loss_gdf

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss # Loss_gdr

            loss = -loss_f + loss_r

        elif self.loss_type == 'ga_klr':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss_f = outputs_f.loss # Loss_gdf

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss # Loss_gdr

            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

            outputs_r_logits = F.log_softmax(outputs_r.logits, dim=-1).view(-1, outputs_r.logits.shape[-1])
            outputs_r_ref_logits = F.log_softmax(outputs_r_ref.logits, dim=-1).view(-1, outputs_r_ref.logits.shape[-1])
            kl_r = F.kl_div(
                outputs_r_logits,
                outputs_r_ref_logits,
                reduction='batchmean',
                log_target=True
            )

            loss = -loss_f + kl_r

        ### npo,npo_gdr,npo_klr
        elif self.loss_type == 'npo':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

            outputs_f_loss = get_batch_loss(outputs_f.logits, x_f['labels'])
            outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_f['labels'])
            neg_log_ratio = outputs_f_loss - outputs_f_ref_loss # neg_log_ratio = -log(P_model(x_f)) - (-log(P_ref(x_f))) = log(P_ref(x_f)) - log(P_model(x_f)) = log(P_ref(x_f) / P_model(x_f))
            loss = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        elif self.loss_type == 'npo_gdr':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

            outputs_f_loss = get_batch_loss(outputs_f.logits, x_f['labels'])
            outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_f['labels'])
            neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
            loss_npo = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta 
            loss = loss_npo + loss_r

        elif self.loss_type == 'npo_klr':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

            outputs_r_logits = F.log_softmax(outputs_r.logits, dim=-1).view(-1, outputs_r.logits.shape[-1])
            outputs_r_ref_logits = F.log_softmax(outputs_r_ref.logits, dim=-1).view(-1, outputs_r_ref.logits.shape[-1])
            kl_r = F.kl_div(
                outputs_r_logits,
                outputs_r_ref_logits,
                reduction='batchmean',
                log_target=True
            )

            outputs_f_loss = get_batch_loss(outputs_f.logits, x_f['labels'])
            outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_f['labels'])
            neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
            loss_npo= -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta 
            loss = loss_npo + kl_r

        ### relearn,relearn_klr,relearn_klr_gdr,relearn_gdr, relearn_dpo, relearn_dpo_gdr, relearn_dpo_klr
        elif self.loss_type in "relearn": # TODO: 增加判断逻辑，对于D_f进行对比学习，对于D_r不变
            assert x_r is None, "retain data is not None but loss type is relearn(gd)."  
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss = outputs_f.loss # Loss_gdf
            
        elif self.loss_type in ["relearn_klr", "relearn_klr_gdr", "relearn_gdr"]:
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss_f = outputs_f.loss # Loss_gdf

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss # Loss_gdr
            
            if self.loss_type == "relearn_gdr":
                loss = loss_f + loss_r
            elif self.loss_type in ["relearn_klr", "relearn_klr_gdr"]:
                with torch.no_grad():
                    outputs_r_ref = self.ref_model(
                        x_r['input_ids'],
                        labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                        attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                    )
                
                outputs_r_logits = F.log_softmax(outputs_r.logits, dim=-1).view(-1, outputs_r.logits.shape[-1])
                outputs_r_ref_logits = F.log_softmax(outputs_r_ref.logits, dim=-1).view(-1, outputs_r_ref.logits.shape[-1])

                kl_r = F.kl_div(
                    outputs_r_logits,
                    outputs_r_ref_logits,
                    reduction='batchmean',
                    log_target=True
                )

                if self.loss_type == "relearn_klr":
                    loss = loss_f + kl_r
                elif self.loss_type == "relearn_klr_gdr":
                    loss = loss_f + kl_r + loss_r
                else:
                    raise NotImplementedError("Cannot infer the given loss type.")
        elif self.loss_type in ["relearn_dpo", "relearn_dpo_gdr", "relearn_dpo_klr"]:
            iwant_outputs = model(
                x_p['input_ids'],
                labels=x_p['labels'] if 'labels' in x_p else x_p['input_ids'].clone(),
                attention_mask=x_p['attention_mask'] if 'attention_mask' in x_p else torch.ones_like(x_p['input_ids'], dtype=torch.bool)
            )
            idontwant_outputs = model(
                x_n['input_ids'],
                labels=x_n['labels'] if 'labels' in x_n else x_n['input_ids'].clone(),
                attention_mask=x_n['attention_mask'] if 'attention_mask' in x_n else torch.ones_like(x_n['input_ids'], dtype=torch.bool)
            )
            with torch.no_grad():
                iwant_outputs_ref = self.ref_model(
                    x_p['input_ids'],
                    labels=x_p['labels'] if 'labels' in x_p else x_p['input_ids'].clone(),
                    attention_mask=x_p['attention_mask'] if 'attention_mask' in x_p else torch.ones_like(x_p['input_ids'], dtype=torch.bool)
                )
                idontwant_outputs_ref = self.ref_model(
                    x_n['input_ids'],
                    labels=x_n['labels'] if 'labels' in x_n else x_n['input_ids'].clone(),
                    attention_mask=x_n['attention_mask'] if 'attention_mask' in x_n else torch.ones_like(x_n['input_ids'], dtype=torch.bool)
                )
                iwant_loss_ref = -1 * iwant_outputs_ref.loss
                idontwant_loss_ref = -1 * idontwant_outputs_ref.loss
            
            iwant_loss = -1 * iwant_outputs.loss
            idontwant_loss = -1 * idontwant_outputs.loss

            pi_logratios = iwant_loss - idontwant_loss
            pi_logratios_ref = iwant_loss_ref - idontwant_loss_ref
            loss = -F.logsigmoid(self.beta * (pi_logratios - pi_logratios_ref)).mean() * 2 / self.beta

            if self.loss_type == "relearn_dpo_gdr":
                retain_outputs = model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )
                loss = loss + retain_outputs.loss
            elif self.loss_type == "relearn_dpo_klr":
                with torch.no_grad():
                    retain_outputs_ref = self.ref_model(
                        x_r['input_ids'],
                        labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                        attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                    )
                retain_probs_ref = F.softmax(retain_outputs_ref.logits, dim=-1).view(-1, retain_outputs_ref.logits.shape[-1])

                retain_outputs = model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )
                retain_probs = F.softmax(retain_outputs.logits, dim=-1).view(-1, retain_outputs.logits.shape[-1])

                retain_loss = F.kl_div(
                    retain_probs,
                    retain_probs_ref,
                    reduction='batchmean',
                    log_target=True
                )

                loss = loss + retain_loss

        # Implement by ysh
        elif self.loss_type in ["contrastive", "contrastive_klr", "contrastive_gdr", "contrastive_klr_gdr"]:
            # 对比学习实现：遗忘精确信息，保留语言能力
            
            # --- 1. 获取问答对的语义嵌入 ---
            outputs_f = model(
                x_f['input_ids'],
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool),
                output_hidden_states=True
            )
            
            # 提取最后一层hidden states
            last_hidden_states = outputs_f.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            
            # 使用答案部分的平均池化作为问答对嵌入
            # 因为我们要学习"忘记精确答案，保留模糊表达"
            if 'attention_mask' in x_f and 'labels' in x_f:
                # 找到答案部分的token位置（labels中非-100的部分）-> 构造数据集时labels对于问题部分进行了掩盖
                answer_mask = (x_f['labels'] != -100).float()  # [batch_size, seq_len]
                attention_mask = x_f['attention_mask'].float()  # [batch_size, seq_len]
                
                # 结合attention_mask和answer_mask，只关注有效的答案token
                valid_answer_mask = answer_mask * attention_mask  # [batch_size, seq_len]
                
                # 对答案部分进行平均池化
                valid_answer_mask_expanded = valid_answer_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
                weighted_hidden = last_hidden_states * valid_answer_mask_expanded  # [batch_size, seq_len, hidden_dim]
                
                # 计算每个样本答案部分的平均嵌入
                sum_embeddings = weighted_hidden.sum(dim=1)  # [batch_size, hidden_dim]
                sum_mask = valid_answer_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
                sum_mask = torch.clamp(sum_mask, min=1e-8)  # 避免除零
                
                embeddings_f = sum_embeddings / sum_mask  # [batch_size, hidden_dim]
            else:
                # 备选方案：使用最后一个非padding token的嵌入
                last_token_indices = x_f['attention_mask'].sum(dim=1) - 1
                batch_indices = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
                embeddings_f = last_hidden_states[batch_indices, last_token_indices, :]
            
            # L2归一化，提高训练稳定性
            embeddings_f = F.normalize(embeddings_f, p=2, dim=1)
            
            # --- 2. 组织样本结构 ---
            # 假设batch组织：[orig_1, var_1_1, var_1_2, var_1_3, orig_2, var_2_1, ...]
            batch_size, hidden_dim = embeddings_f.shape
            if batch_size % 4 != 0:
                raise ValueError("批次大小必须是4的倍数 (1个原始 + 3个变体)")
            
            num_groups = batch_size // 4
            grouped_embeddings = embeddings_f.view(num_groups, 4, hidden_dim)
            
            # 分离原始问答（负样本）和变体问答（正样本池）
            original_embeddings = grouped_embeddings[:, 0, :]  # [num_groups, hidden_dim] - 精确答案
            variant_embeddings = grouped_embeddings[:, 1:, :]   # [num_groups, 3, hidden_dim] - 模糊变体
            
            # --- 3. 实现InfoNCE对比学习损失 ---
            temperature = 0.07  # 温度系数
            total_contrastive_loss = 0.0
            
            # 策略：每个变体作为锚点，学习远离原始精确答案，靠近其他变体
            for anchor_idx in range(3):  # 3个变体轮流作为锚点
                anchors = variant_embeddings[:, anchor_idx, :]  # [num_groups, hidden_dim]
                
                # 正样本：同一组的其他变体
                positive_indices = [i for i in range(3) if i != anchor_idx]
                positives = variant_embeddings[:, positive_indices, :].reshape(-1, hidden_dim)  # [num_groups*2, hidden_dim]
                
                # 负样本：原始精确答案 + 其他组的所有样本
                negatives_list = []
                
                # 1) 当前组的原始答案（主要负样本）
                negatives_list.append(original_embeddings)  # [num_groups, hidden_dim]
                
                # 2) 其他组的所有样本（次要负样本，增加难度）
                other_groups_samples = []
                for group_idx in range(num_groups):
                    # 排除当前组
                    other_group_indices = [i for i in range(num_groups) if i != group_idx]
                    if other_group_indices:
                        other_samples = grouped_embeddings[other_group_indices].reshape(-1, hidden_dim)
                        other_groups_samples.append(other_samples)
                
                if other_groups_samples:
                    all_other_samples = torch.cat(other_groups_samples, dim=0)
                    # 随机采样，避免负样本过多
                    sample_size = min(num_groups * 2, all_other_samples.size(0))
                    indices = torch.randperm(all_other_samples.size(0))[:sample_size]
                    sampled_others = all_other_samples[indices]
                    negatives_list.append(sampled_others)
                
                negatives = torch.cat(negatives_list, dim=0)  # [total_negatives, hidden_dim]
                
                # 计算相似度分数
                for group_idx in range(num_groups):
                    anchor = anchors[group_idx:group_idx+1]  # [1, hidden_dim]
                    
                    # 当前组的正样本
                    group_positives = variant_embeddings[group_idx, positive_indices, :]  # [2, hidden_dim]
                    
                    # 计算anchor与正样本的相似度
                    pos_sim = F.cosine_similarity(anchor, group_positives, dim=1)  # [2]
                    
                    # 计算anchor与负样本的相似度  
                    neg_sim = F.cosine_similarity(anchor, negatives, dim=1)  # [total_negatives]
                    
                    # InfoNCE损失：对于每个正样本分别计算
                    for pos_idx in range(len(pos_sim)):
                        # 将当前正样本与所有负样本组合
                        logits = torch.cat([
                            pos_sim[pos_idx:pos_idx+1],  # 当前正样本
                            neg_sim  # 所有负样本
                        ]) / temperature
                        
                        # 标签：正样本在第0位
                        labels = torch.zeros(1, device=logits.device, dtype=torch.long)
                        
                        # 计算交叉熵损失
                        total_contrastive_loss += F.cross_entropy(logits.unsqueeze(0), labels)
            
            # 对3个锚点的损失求平均
            loss_con = total_contrastive_loss / 3
            loss = loss_con
            
            # --- 4. 结合保留数据损失 ---
            if self.loss_type == 'contrastive':
                loss = loss_con
                
            elif self.loss_type == 'contrastive_gdr':
                outputs_r = model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )
                loss_r = outputs_r.loss # Loss_gdr
                loss = loss_con + loss_r
                
            elif self.loss_type in ['contrastive_klr', 'contrastive_klr_gdr']:
                outputs_r = model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )
                loss_r = outputs_r.loss # Loss_gdr
                
                with torch.no_grad():
                    outputs_r_ref = self.ref_model(
                        x_r['input_ids'],
                        labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                        attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                    )
                
                outputs_r_logits = F.log_softmax(outputs_r.logits, dim=-1).view(-1, outputs_r.logits.shape[-1])
                outputs_r_ref_logits = F.log_softmax(outputs_r_ref.logits, dim=-1).view(-1, outputs_r_ref.logits.shape[-1])
                kl_r = F.kl_div(
                    outputs_r_logits,
                    outputs_r_ref_logits,
                    reduction='batchmean',
                    log_target=True
                )
                 
                if self.loss_type == 'contrastive_klr':
                    loss = loss_con + kl_r
                elif self.loss_type == 'contrastive_klr_gdr':
                    loss = loss_con + kl_r + loss_r
                else:
                    raise NotImplementedError("Cannot infer the given loss type.")
                
            else:
                raise NotImplementedError("Cannot infer the given loss type.")
            
            # # 返回合适的outputs对象
            # if 'outputs_r' in locals():
            #     outputs_to_return = outputs_r
            # else:
            #     outputs_to_return = outputs_f            
        
        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        return (loss, outputs_f) if return_outputs else loss

    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
