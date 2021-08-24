from grb.dataset import Dataset

if __name__ == '__main__':
    # Load data
    dataset_name = 'grb-cora'
    dataset = Dataset(name=dataset_name,
                      data_dir="../data/",
                      mode='full',
                      feat_norm='arctan')

    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    test_mask = dataset.test_mask

    # Load model
    from grb.model.torch import GCN
    from grb.utils.normalize import GCNAdjNorm

    model_name = "gcn"
    model_sur = GCN(in_features=dataset.num_features,
                    out_features=dataset.num_classes,
                    hidden_features=64,
                    n_layers=2,
                    adj_norm_func=GCNAdjNorm,
                    layer_norm=False,
                    residual=False,
                    dropout=0.5)
    print(model_sur)

    # Prepare attack
    from grb.attack.injection.tdgia import TDGIA

    device = 'cuda:0'

    attack = TDGIA(lr=0.001,
                   n_epoch=100,
                   n_inject_max=100,
                   n_edge_max=200,
                   feat_lim_min=-0.99,
                   feat_lim_max=0.99,
                   device=device,
                   inject_mode='tdgia')

    adj_attack, features_attack = attack.attack(model=model_sur,
                                                adj=adj,
                                                features=features,
                                                target_mask=test_mask)
    print(adj_attack, features_attack)
