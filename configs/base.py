from easydict import EasyDict as edict


config = edict()
config.output = None

#Dataset properties:
config.path_to_dataset = '/Users/robinin/face_recognition/datasets/renamed_clean'
config.nir_folder = 'NIR'
config.vis_folder = 'VIS'
config.load_same_label_pairs = './datasets/same_label_pairs.json'
config.save_same_label_pairs = './datasets/same_label_pairs.json'
config.load_subindex_for_label = './datasets/subindex_for_label.json'   #'./datasets/subindex_for_label.json'
config.save_subindex_for_label = './datasets/subindex_for_label.json'

#Training
config.n_epoch = 20
config.batch_size = 64
config.learning_rate = .01

#PDT
config.PDT_pool_features = 6

#Loss function
config.pretrained_on = 'vggface2'
config.loss_margin = 2

#PDT

