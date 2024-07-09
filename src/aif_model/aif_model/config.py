# Image config
height = 32
width = 32
channels = 3
horizontal_fov = 1.3963

# Training config
latent_size = 6
learning_rate = 1e-3
variance = 1
n_epochs = 100
n_batch = 512
beta = 2 # for betaVAE

# Agent config
vae_path = "/home/tin/FER/Diplomski/4.semestar/Diplomski rad/code/src/aif_model/resource/vae_state_dict.pt"
intentions_path = "/home/tin/FER/Diplomski/4.semestar/Diplomski rad/code/src/aif_model/resource/intentions_state_dict.pt"
focus_samples = ["/home/tin/FER/Diplomski/4.semestar/Diplomski rad/code/src/aif_model/resource/focus0.csv","/home/tin/FER/Diplomski/4.semestar/Diplomski rad/code/src/aif_model/resource/focus1.csv"]#,"/home/tin/FER/Diplomski/4.semestar/Diplomski rad/code/src/aif_model/resource/focus2.csv","/home/tin/FER/Diplomski/4.semestar/Diplomski rad/code/src/aif_model/resource/focus3.csv"]
n_orders = 2 # orders of belief
num_intentions = 2
prop_len = 2 # size of proprioceptive belief
needs_len = 2 # size of needs belief
k = 0.06
alpha = 0.5
pi_prop = 0.5
pi_need = 0.5
pi_vis = 6e-3
dt = 0.4
a_max = 2.0

limits = [[90,90],[-90,-90]] # [[pitch min, yaw min],[pitch max, yaw max]]
noise = 5e-5 # action noise

update_frequency = 20
