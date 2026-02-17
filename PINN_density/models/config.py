TEST_CONFIG = {
    'n_samples': 2000,
    'random_seed': 42,
    'dopant_types': ['Ca', 'Sr', 'Mg'],
    'feature_ranges': {
        'T_K': (300, 1200, 10),
        'xA_mol': (0, 0.5, 10),
        'xB_mol': (0, 0.5, 10),
        'xV_mol': (0, 0.3, 10),
    },
    'target_column': 'log(Conductivity(S/cm))'
}