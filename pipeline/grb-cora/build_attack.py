def build_attack(attack_name, dataset, device="cpu", args=None):
    if attack_name in "rnd":
        from grb.attack.rnd import RND

        config = {}
        config['feat_lim_min'] = args.feat_lim_min
        config['feat_lim_max'] = args.feat_lim_max
        config['n_inject_max'] = args.n_inject
        config['n_edge_max'] = args.n_edge_max

        attack = RND(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)
    elif attack_name in "fgsm":
        from grb.attack.fgsm import FGSM

        config = {}
        config['epsilon'] = args.lr
        config['n_epoch'] = args.n_epoch
        config['feat_lim_min'] = args.feat_lim_min
        config['feat_lim_max'] = args.feat_lim_max
        config['n_inject_max'] = args.n_inject
        config['n_edge_max'] = args.n_edge_max

        attack = FGSM(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)
    elif attack_name in "pgd":
        from grb.attack.pgd import PGD

        config = {}
        config['epsilon'] = args.lr
        config['n_epoch'] = args.n_epoch
        config['feat_lim_min'] = args.feat_lim_min
        config['feat_lim_max'] = args.feat_lim_max
        config['n_inject_max'] = args.n_inject
        config['n_edge_max'] = args.n_edge_max

        attack = PGD(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)
    elif attack_name in "speit":
        from grb.attack.speit import SPEIT

        config = {}
        config['inject_mode'] = 'random'
        config['lr'] = args.lr
        config['n_epoch'] = args.n_epoch
        config['feat_lim_min'] = args.feat_lim_min
        config['feat_lim_max'] = args.feat_lim_max
        config['n_inject_max'] = args.n_inject
        config['n_edge_max'] = args.n_edge_max

        attack = SPEIT(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)

        attack = SPEIT(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)
    elif attack_name in "tdgia":
        from grb.attack.tdgia import TDGIA

        config = {}
        config['inject_mode'] = 'random'
        config['lr'] = args.lr
        config['n_epoch'] = args.n_epoch
        config['feat_lim_min'] = args.feat_lim_min
        config['feat_lim_max'] = args.feat_lim_max
        config['n_inject_max'] = args.n_inject
        config['n_edge_max'] = args.n_edge_max

        attack = TDGIA(dataset, adj_norm_func=None, device=device)
        attack.set_config(**config)

    return attack
