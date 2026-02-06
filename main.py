import argparse
import os
import yaml
import torch
from exp.exp_ppo import Exp_PPO

import logging

def setup_logging(res_path):
    # 核心修复：确保日志所在的目录已经创建
    log_dir = res_path 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    log_file = os.path.join(log_dir, 'training_log.txt')
    
    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():


    parser = argparse.ArgumentParser(description='PPO for ZZ500 Timing')
    
    # 1. 基础路径配置
    parser.add_argument('--data_path', type=str, default='/cpfs/dss/dev/gzyu/RL_Optimization_of_asset_strategies/RL_mult_asset/data/daily_2010_2024pricemain_cta.feather', help='数据文件路径')
    parser.add_argument('--save_dir', type=str, default='./results_ent/', help='实验结果保存根目录')
    parser.add_argument('--exp_name', type=str, default='timing_v1', help='实验名称')
    
    # 2. 从 yaml 加载默认参数
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 3. 策略与环境参数 (允许命令行覆盖)
    parser.add_argument('--min_pos', type=float, default=config.get('min_pos', 0.0))
    parser.add_argument('--max_pos', type=float, default=config.get('max_pos', 1.0))
    parser.add_argument('--commission', type=float, default=config.get('commission', 0.0003))
    parser.add_argument('--fix_weight', type=float, default=config.get('fix_weight', 1.0))

    # 4. 训练与模型参数 
    parser.add_argument('--lr', type=float, default=config.get('lr', 0.0001))
    parser.add_argument('--gamma', type=float, default=config.get('gamma', 0.99))
    parser.add_argument('--gae_lambda', type=float, default=config.get('gae_lambda', 0.95))
    parser.add_argument('--clip_eps', type=float, default=config.get('clip_eps', 0.2))
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 64))
    parser.add_argument('--n_epochs', type=int, default=config.get('n_epochs', 10))
    parser.add_argument('--ppo_epochs', type=int, default=config.get('ppo_epochs', 10))
    parser.add_argument('--buffer_size', type=int, default=config.get('buffer_size', 1024))
    parser.add_argument('--hidden_dim', type=int, default=config.get('hidden_dim', 128))
    parser.add_argument('--n_layers', type=int, default=config.get('n_layers', 2))

    parser.add_argument('--vf_coef', type=float, default=config.get('vf_coef', 0.95))
    parser.add_argument('--ent_coef', type=float, default=config.get('ent_coef', 0.03))

    parser.add_argument('--alpha_init', type=float, default=config.get('alpha_init', 10.0))
    parser.add_argument('--beta_init', type=float, default=config.get('beta_init', 10.0))
    parser.add_argument('--window_size', type=int, default=config.get('window_size', 20))
    parser.add_argument('--risk_beta', type=float, default=config.get('risk_beta', 0.2))

    parser.add_argument('--n_assets', type=int, default=config.get('n_assets', 66))
    parser.add_argument('--feature_dim', type=int, default=config.get('feature_dim', 9))

    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用 GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 设备 ID')

    args = parser.parse_args()
    
    # 5. 设备配置
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
        print(f"检测到 GPU，正在使用: {torch.cuda.get_device_name(0)}")
    else:
        args.device = torch.device("cpu")
        print("未检测到 GPU 或未开启 GPU 选项，正在使用 CPU")

    # 6. 创建实验文件夹结构
    # 最终结果会存放在 ./experiments/timing_v1/ 目录下
    


    

    for args.lr in [0.0001,0.00003,0.00001]:
        for args.batch_size in [64,80,96]:
            #for args.buffer_size in [512,1024,256]:
                #for args.hidden_dim in [128,256,512]:
                    #for args.n_layers in [2,3,4]:
                        
                        exp_des = f"lr{args.lr}_batch{args.batch_size}_buffer{args.buffer_size}_hidden{args.hidden_dim}_layers{args.n_layers}"
                                        

                        args.res_path = os.path.join(args.save_dir, exp_des)
                        args.checkpoints = os.path.join(args.res_path, 'checkpoints')
                        args.results_dir = os.path.join(args.res_path, 'results')

                        os.makedirs(args.res_path, exist_ok=True)
                        setup_logging(args.res_path)
                        logging.info(f"Starting experiment: {args.exp_name}")

                        os.makedirs(args.checkpoints, exist_ok=True)
                        os.makedirs(args.results_dir, exist_ok=True)

                        print("="*50)
                        print(f"Starting experiment: {args.exp_name}")
                        print(f"Device: {args.device}")
                        print(f"Data: {args.data_path}")
                        print(f"Baseline Weight: {args.fix_weight}")

                        print(f"args:{args}")
                        print("="*50)

                        exp = Exp_PPO(args)
                        
                        print("\n>>> Training")
                        exp.train()
                        

                        print("\n>>> Testing & Evaluation")
                        exp.test()
                        
                        print(f"\nExperiment finished. Results saved in: {args.res_path}")
    


if __name__ == "__main__":
    main()
