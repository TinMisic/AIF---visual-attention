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
package_name = "aif_model"
vae_path = "vae-disentangled_state_dict_scaled.pt"
focus_samples = ["focus0.csv","focus1.csv"]#,"focus2.csv","focus3.csv"]
n_orders = 2 # orders of belief
num_intentions = 2
prop_len = 2 # size of proprioceptive belief
needs_len = 3 # size of needs/cueing belief
focus_len = 3 # size of focus belief: amplitude, x_position, y_position
k = 0.06
alpha = 0.5
pi_prop = 0.5
pi_need = 0.5
pi_vis =  7e-3#7e-3
foveation_sigma = 2
dt = 0.4 # 0.4
a_max = 1.0

limits = [[90,90],[-90,-90]] # [[pitch min, yaw min],[pitch max, yaw max]]
noise = 5e-5 # action noise

update_frequency = 20

def set_pi_vis(value):
    global pi_vis
    pi_vis = value