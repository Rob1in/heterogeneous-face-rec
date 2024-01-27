from easydict import EasyDict as edict


config = edict()
config.output = None

#Profiler
config.profile = True
config.profiler_logs_path = './logs/profiles/profile.log'

#Dataset properties:
config.path_to_dataset = '/Users/robinin/face_recognition/datasets/renamed_clean'
config.nir_folder = 'NIR'
config.vis_folder = 'VIS'
config.load_same_label_pairs = './datasets/same_label_pairs.json'
config.save_same_label_pairs = './datasets/same_label_pairs.json'
config.load_subindex_for_label = './datasets/subindex_for_label.json'   #'./datasets/subindex_for_label.json'
config.save_subindex_for_label = './datasets/subindex_for_label.json'

#Dataloader properties:
config.dl_num_workers = 4

#Training
config.n_epoch = 1
config.batch_size = 64
config.learning_rate = .001

#PDT
config.PDT_pool_features = 6

#Loss function
config.pretrained_on = 'vggface2'
config.loss_margin = 2

#PDT

