
from splits import get_splits

class Config():
    def __init__(self, dataset='mnist', split_num=0, latent_dims=128, fixed_dims=0, lv=False):
        self.data_base_path = './data'
        self.val_ratio = 0.2
        self.seed = 1234
        self.known_classes = ''
        self.unknown_classes = ''
        self.split_num = split_num
        self.batch_size = 32
        self.num_workers = 21
        self.dataset = dataset
        self.epochs = 100

        self.z_dim = latent_dims
        self.fixed_dim = fixed_dims
        self.learned_variance = lv

        self.lr = 0.00001
        self.lr_step = 10
        self.lr_gamma = 0.9

        self.conv_layers = [[58,4,2,1],[72,4,2,1],[112,4,1,0]]
        self.linear_layers = [88]

        self.loss_KL = 5/6
        self.loss_rec = 1/6
        self.loss_cluster = 0
        self.loss_radial = 0

        self.checkpoint = ''
        self.save_name = ''

        # Process args
        if self.dataset != "": self.process_args()

    def base_path(self):
        return f'runs/{self.dataset}/{self.fixed_dim}_{self.z_dim-self.fixed_dim}_{self.split_num}'

    def process_args(self):
        known_classes = [int(cl) for cl in self.known_classes.split()]
        unknown_classes = [int(cl) for cl in self.unknown_classes.split()]
        splits = get_splits(self.dataset, num_split=self.split_num)
        self.known_classes = known_classes if len(known_classes) > 0 else splits['known_classes']
        self.unknown_classes = unknown_classes if len(unknown_classes) > 0 else splits['unknown_classes']
