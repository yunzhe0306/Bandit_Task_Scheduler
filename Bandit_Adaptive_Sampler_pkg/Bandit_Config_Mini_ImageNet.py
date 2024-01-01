import argparse


def get_bandit_sampler_config():
    parser = argparse.ArgumentParser(description='Bandit')

    parser.add_argument('--hidden_size', default=200, type=int, help='hidden size for bandit sampler')
    parser.add_argument('--lr_rate', default=0.01, type=float, help='learning rate for bandit sampler')

    parser.add_argument('--g_pool_step_meta', default=300, type=int, help='pooling step for meta-model')
    parser.add_argument('--g_pool_step_bandit', default=400, type=int, help='pooling step for bandit sampler')

    parser.add_argument('--buffer_size', default=300, type=int, help='buffer size for received records')
    parser.add_argument('--alpha', default=0.5, type=float, help='coefficient for balancing two exploration objectives')

    parser.add_argument('--train_per_step', default=10, type=int, help='steps per training operation')

    #######
    parser.add_argument('--artificial_exp_score', default=0, type=float, help='Artificial exploration score for f_2')
    #######

    args = parser.parse_args()

    print("\n\n------------ Bandit Args")
    print(args)

    return args
