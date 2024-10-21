import aif_model.vae as vae
import aif_model.config as c
import aif_model.networks as networks
import aif_model.utils as utils
import torch
import numpy as np
import cv2
from ament_index_python.packages import get_package_share_directory
import os

class Agent:
    """
    Active Inference agent
    """
    def __init__(self):

        # Load networks
        package_share_directory = get_package_share_directory(c.package_name)
        vae_path = os.path.join(package_share_directory, 'resource', c.vae_path)

        self.vectors = np.zeros((2,2)) # vectors of target locations from center

        self.vae  = vae.VAE( # VAE network
        latent_dim=c.latent_size,
        encoder=networks.Encoder(in_chan=c.channels, latent_dim=c.latent_size),
        decoder=networks.Decoder(out_chan=c.channels, latent_dim=c.latent_size))
        self.vae.load(vae_path)

        self.belief_dim = c.needs_len + c.prop_len + c.latent_size # needs, proprioceptive belief, visual belief, visual intentions

        # Initialization of variables
        self.mu = np.zeros((c.n_orders, self.belief_dim), dtype="float32") 
        self.mu_dot = np.zeros_like(self.mu)
        self.focus_samples = []

        self.a = np.zeros(c.prop_len)
        self.a_dot = np.zeros_like(self.a)

        self.E_i = np.zeros((c.num_intentions, self.belief_dim))

        self.alpha = np.array([1, 1, 1]) # needs, proprioceptive, visual [1, c.alpha, 1-c.alpha]

        self.beta_index = 0
        weights = [] 
        for i in range(c.num_intentions):
            builder = np.array([1]*c.needs_len+[1]*c.prop_len+[1e-1]*c.latent_size)
            weights.append(builder)

        self.beta_weights = weights
        self.mode = "closest"

        # Generative models (simple)
        self.G_p = utils.shift_rows(np.eye(self.belief_dim, c.prop_len),c.needs_len)

        self.G_n = np.eye(self.belief_dim, c.needs_len)

    def get_p(self):
        """
        Get predicitons
        """
        # Visual
        input_, output = self.vae.predict_visual(torch.tensor(self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.latent_size]).unsqueeze(0))
        # Proprioceptive
        p_prop = self.mu[0].dot(self.G_p)
        # Needs
        p_needs = self.mu[0].dot(self.G_n)

        P = [p_needs, p_prop, output.detach().squeeze().cpu().numpy()]

        return P, [input_, output]
    
    def get_vis_intentions(self):
        targets_vis = []
        current_mu = self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.latent_size]
        for i in range(c.num_intentions): 
            if self.mode == "closest":# find closest focus sample for each intention
                diff = np.linalg.norm((self.focus_samples[i]-current_mu),axis=1) # euclidean distance
                closest = np.argmin(diff)
                targets_vis.append(self.focus_samples[i][closest])
                #print("Focus sample id for object "+str(i)+" is "+str(closest)+" , diff is",diff[closest])
            elif self.mode == "mean":
                targets_vis.append(np.mean(self.focus_samples[i],axis=0))

        return np.array(targets_vis)
    
    def get_i(self):
        """
        Get intentions
        """
        targets = self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.prop_len*c.num_intentions] # grab visual positions of objects
        targets = np.reshape(targets,(c.num_intentions,c.prop_len)) # reshape
        targets = utils.denormalize(targets) # convert from range [-1,1] to [0,width]
        # print("Target in pixels:",targets)
        self.vectors = np.array(targets)
        targets = utils.pixels_to_angles(targets) # convert to angles

        targets_prop = targets + self.mu[0,c.needs_len:c.needs_len+c.prop_len] # add relative target angle to global camera angle
        targets_vis = self.get_vis_intentions()
        targets_needs = np.tile(self.mu[0,:c.needs_len],(c.num_intentions,1))

        ending = np.tile(self.mu[0,c.needs_len+c.prop_len+c.latent_size:],(c.num_intentions,1)) # get ending of intention vectors from belief
        targets = np.concatenate((targets_needs,targets_prop,targets_vis,ending),axis=1) # concatenate to get final matrix of shape NUM_INTENTIONS x (PROP_LEN + DESIRE_LEN + LATENT_SIZE)

        return targets
    
    def get_e_s(self, S, P):
        """
        Get sensory prediction errors
        """
        return [s - p for s, p in zip(S, P)]
    
    def get_e_mu(self, I):
        """
        Get dynamics prediction errors
        """
        self.E_i = (I - self.mu[0]) * c.k * 1 # self.mode = 1 for now

        return self.mu[1] - self.E_i
    
    def get_sensory_precisions(self, S):
        # TODO: implement attention map
        Pi = list()
        Pi.append(np.ones(c.needs_len+c.prop_len+c.latent_size) * c.pi_need)
        Pi.append(np.ones(c.needs_len+c.prop_len+c.latent_size) * c.pi_prop)
        Pi.append(utils.pi_foveate(np.ones((c.height,c.width)) * c.pi_vis))

        return Pi
    
    def get_intention_precisions(self, S):
        self.beta_index = np.argmax(self.mu[0,:c.needs_len])
        self.beta = [np.ones(c.needs_len+c.prop_len+c.latent_size)*1e-10] * c.num_intentions
        self.beta[self.beta_index] = self.beta_weights[self.beta_index]
        return self.beta
    
    def get_likelihood(self, E_s, grad_v, Pi):
        """
        Get likelihood components
        """
        lkh = {}
        lkh['need'] = self.alpha[0] * Pi[0] * E_s[0].dot(self.G_n.T)

        lkh['prop'] = self.alpha[1] * Pi[1] * E_s[1].dot(self.G_p.T)

        lkh['vis'] = self.alpha[2] * self.vae.get_grad(*grad_v, torch.from_numpy(Pi[2])*E_s[2])
        lkh['vis'] = np.concatenate((np.zeros((c.needs_len+c.prop_len)),lkh['vis'])) 

        return lkh
    
    def get_precision_derivatives_mu(self, Pi, Gamma):
        # First order
        dPi_dmu0 = [np.zeros((self.belief_dim,self.belief_dim)), np.zeros((self.belief_dim,self.belief_dim)), np.zeros((c.height,c.width,self.belief_dim))] # TODO: Finish

        dGamma_dmu0 = [np.zeros((self.belief_dim, self.belief_dim))] * c.num_intentions # TODO: Finish

        # Second order
        dPi_dmu1 = [np.zeros((self.belief_dim,self.belief_dim)), np.zeros((self.belief_dim,self.belief_dim)), np.zeros((c.height,c.width,self.belief_dim))] # TODO: Finish

        dGamma_dmu1 = [np.zeros((self.belief_dim, self.belief_dim))] * c.num_intentions # TODO: Finish

        return dPi_dmu0, dGamma_dmu0, dPi_dmu1, dGamma_dmu1


    def attention(self, precision, derivative, error):
        total = np.zeros(self.belief_dim)
        for i in range(len(precision)):
            component1 = 0.5 * np.sum(np.expand_dims(1/precision[i],axis=-1) * derivative[i], axis=tuple(range(derivative[i].ndim - 1)))
            component2 = -0.5 * np.sum(np.expand_dims(error[i]**2, axis=-1) * derivative[i], axis=tuple(range(derivative[i].ndim - 1)))
            total += component1 + component2

        return total

    def get_mu_dot(self, lkh, E_s, E_mu, Pi, Gamma):
        """
        Get belief update
        """
        self.mu_dot = np.zeros_like(self.mu)

        # Pad needs and proprioceptive error to size of mu
        e_s = [np.concatenate([E_s[0],np.zeros(self.belief_dim - c.needs_len)]),np.concatenate([E_s[1],np.zeros(self.belief_dim - c.prop_len)]), torch.mean(E_s[2],dim=(0,1))]

        # Intention components
        forward_i = np.zeros((self.belief_dim)) 
        for g, e in zip(Gamma, np.array(E_mu)):
            forward_i += g * e

        dPi_dmu0, dGamma_dmu0, dPi_dmu1, dGamma_dmu1 = self.get_precision_derivatives_mu(Pi, Gamma)

        generative = lkh['prop'] + lkh['need'] + lkh['vis']
        backward = - c.k * forward_i

        bottom_up0 = self.attention(Pi,dPi_dmu0,e_s)
        top_down0 = self.attention(Gamma,dGamma_dmu0,E_mu)

        bottom_up1 = self.attention(Pi, dPi_dmu1,[0]*3) # No sensory error for second order
        top_down1 = self.attention(Gamma,dGamma_dmu1,[0]*c.num_intentions) # No intention error for second order

        self.mu_dot[0] = self.mu[1] + generative + backward + bottom_up0 + top_down0
        self.mu_dot[1] = -forward_i + bottom_up1 + top_down1

    def get_a_dot(self, likelihood, Pi):
        """
        Get action update
        """
        e_prop = likelihood["prop"].dot(self.G_p)

        # lkh_vis= np.array(likelihood["vis"][c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.prop_len*c.num_intentions], dtype="float32")
        # lkh_vis = np.reshape(lkh_vis,(c.num_intentions,c.prop_len))
        # lkh_pix = utils.denormalize(lkh_vis)
        # lkh_ang = utils.pixels_to_angles(lkh_pix)

        # index = np.argmax(np.linalg.norm(lkh_angles,axis=1))# max = biggest surprise; 
        # index = self.beta_index#np.argmax(self.beta) # from beta = most desired; 
        
        # lkh_angles = lkh_ang[index]
        # lkh_angles = lkh_angles[index]
        # old_lkh_angles = np.mean(old_lkh_ang,axis=0)# avg = average movement

        # d_mu_lkh_vis = c.dt * lkh_angles
        d_mu_lkh_prop = -c.dt * e_prop

        # print("dmu_lkh_vis",d_mu_lkh_vis)
        # print("dmu_lkh_prop",d_mu_lkh_prop)

        self.a_dot = d_mu_lkh_prop #c.alpha * d_mu_lkh_prop + (1 - c.alpha) * d_mu_lkh_vis
        print("a_dot",self.a_dot)

    def integrate(self):
        """
        Integrate with gradient descent
        """
        if (self.mu_dot == np.nan).any():
            print("nan in mu_dot",self.mu_dot)
            raise Exception("nan in mu_dot")
        # Update belief
        self.mu[0] += c.dt * self.mu_dot[0]
        self.mu[1] += c.dt * self.mu_dot[1]

        # Update action
        self.a += c.dt * self.a_dot
        self.a = np.clip(self.a, -c.a_max, c.a_max)

    def init_belief(self, needs, prop, visual):
        """
        Initialize belief
        """
        visual_state =self.vae.predict_latent(visual.squeeze()).detach().squeeze().numpy()

        package_share_directory = get_package_share_directory(c.package_name)
        for f_name in [os.path.join(package_share_directory, 'resource', x) for x in c.focus_samples]:
            focus_samples = np.loadtxt(f_name,delimiter=",",dtype="float32")
            self.focus_samples.append(focus_samples)
        self.mu[0] = np.concatenate((needs, prop, visual_state)) # initialize with beliefs about needs, proprioceptive and visual state
        print("mu initialized to:",self.mu[0])

        self.beta_index = np.argmax(needs)
        self.beta = np.ones((c.num_intentions,c.needs_len+c.prop_len+c.latent_size))*1e-10
        self.beta[self.beta_index] = self.beta_weights[self.beta_index]


    def switch_mode(self, step):
        if step%5== 0:
            # self.mode="mean"
            if self.mode=="closest": self.mode = "mean"
            elif self.mode=="mean": self.mode = "closest"
        # else:
        #     self.mode = "closest"
        

    def inference_step(self, S, step):
        """
        Run an inference step
        """
        # TODO: remove
        if (self.mu == np.nan).any():
            print("nan in mu",self.mu)
            print("S",S)
            raise Exception("nan in mu")
        
        # Get predictions
        P, grad_v = self.get_p()

        # Get intentions
        I = self.get_i()

        # Get sensory prediction errors
        E_s = self.get_e_s(S, P)

        # Get dynamics prediction errors
        E_mu = self.get_e_mu(I)

        # Get sensory precisions
        Pi = self.get_sensory_precisions(S)

        # Get intention precisions
        Gamma = self.get_intention_precisions(S)

        # Get likelihood components
        likelihood = self.get_likelihood(E_s, grad_v, Pi)

        # Get belief update
        self.get_mu_dot(likelihood, E_s, E_mu, Pi, Gamma)

        # Get action update
        self.get_a_dot(likelihood, Pi) # E_s[0] * self.pi_s[0] * self.alpha[0]

        # Update
        self.integrate()

        # Show visual sensory and predicted data
        utils.show_SP(S, P, self.vectors)

        # Start action
        self.switch_mode(step)

        return self.a
