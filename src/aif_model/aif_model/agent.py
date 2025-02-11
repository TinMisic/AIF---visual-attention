import aif_model.vae as vae
import aif_model.config as c
import aif_model.networks as networks
import aif_model.utils as utils
import torch
import numpy as np
import cv2
from ament_index_python.packages import get_package_share_directory
import os
from scipy.special import softmax

class Agent:
    """
    Active Inference agent
    """
    def __init__(self):

        # Load networks
        package_share_directory = get_package_share_directory(c.package_name)
        vae_path = os.path.join(package_share_directory, 'resource', c.vae_path)

        self.vectors = np.zeros((3,2)) # vectors of target locations from center, and focus point

        self.vae  = vae.VAE( # VAE network
        latent_dim=c.latent_size,
        encoder=networks.Encoder(in_chan=c.channels, latent_dim=c.latent_size),
        decoder=networks.Decoder(out_chan=c.channels, latent_dim=c.latent_size))
        self.vae.load(vae_path)

        self.belief_dim = c.needs_len + c.prop_len + c.latent_size + c.focus_len # needs, proprioceptive belief, visual belief, visual focus 

        # Initialization of variables
        self.mu = np.zeros((c.n_orders, self.belief_dim), dtype="float32") 
        self.mu_dot = np.zeros_like(self.mu)

        self.a = np.zeros(c.prop_len)
        self.a_dot = np.zeros_like(self.a)

        self.E_i = np.zeros((c.num_intentions, self.belief_dim))

        self.alpha = np.array([1, 1, 1]) # needs, proprioceptive, visual [1, c.alpha, 1-c.alpha]

        weights = [] 
        for i in range(c.num_intentions):
            builder = np.array([1]*c.needs_len+[1]*c.prop_len+[1e-1]*c.latent_size+[1]*c.focus_len)
            weights.append(builder)

        self.beta_weights = np.array(weights)
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
    
    def get_prop_intentions(self):
        targets = np.zeros((c.num_intentions,c.prop_len))
        targets = self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+(2+1)*c.num_intentions] # grab visual positions of objects
        targets = np.reshape(targets,(c.num_intentions,2+1))[:,:2] # reshape and cut
        print("Targets in latent:",targets)
        targets = utils.denormalize(targets) # convert from range [-1,1] to [0,width]
        print("Target in pixels:",targets)
        self.vectors[:2,:] = np.array(utils.normalize(targets))
        targets = utils.pixels_to_angles(targets) # convert to angles

        result = np.zeros_like(targets) + self.mu[0,c.needs_len:c.needs_len+c.prop_len]
        if self.mu[0, c.needs_len+c.prop_len+c.prop_len]>0: # if red object visible in image
            result[0] += targets[0] # add relative target angle to global camera angle
        if self.mu[0, c.needs_len+c.prop_len+c.prop_len+1+c.prop_len]>0: # if blue object visible in image
            result[1] += targets[1] # add relative target angle to global camera angle

        return result
    
    def get_vis_intentions(self):
        red_cue = self.mu[0,2]
        red_exist = self.mu[0, c.needs_len+c.prop_len+c.prop_len]
        sm_r = softmax((red_cue,red_exist))

        blue_cue = self.mu[0,5]
        blue_exist = self.mu[0, c.needs_len+c.prop_len+c.prop_len+1+c.prop_len]
        sm_b = softmax((blue_cue,blue_exist))
        
        old_r = self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.prop_len]
        cue_r = self.mu[0,:c.prop_len]
        old_b = self.mu[0,c.needs_len+c.prop_len+c.prop_len+1:c.needs_len+c.prop_len+c.prop_len+1+c.prop_len]
        cue_b = self.mu[0,c.prop_len+1:c.prop_len+1+c.prop_len]

        visual = self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.latent_size]
        mix_r = sm_r[0]*old_r + sm_r[1]*cue_r
        mix_b = sm_b[0]*old_b + sm_b[1]*cue_b

        result = np.zeros((c.num_intentions,c.latent_size))

        red_target = np.copy(visual)
        red_target[:2] = mix_r
        result[0]=red_target

        blue_target = np.copy(visual)
        blue_target[3:5] = mix_b
        result[1]=blue_target

        return result

    def get_focus_intentions(self):
        result = np.zeros((c.num_intentions,c.focus_len))
        result[:,1:] = self.mu[0,-2:] # previous focus belief
        if self.mu[0,2]>0.1:
            result[0,1:] = self.mu[0,:2]
        if self.mu[0,5]>0.1:
            result[1,1:] = self.mu[0,3:5]

        amp = self.mu[0,c.needs_len+c.prop_len+c.latent_size]
        result[0,0] = 0.1*self.mu[0,2] - 0.5*amp
        result[1,0] = 0.1*self.mu[0,5] - 0.5*amp 

        print("Focus intentions:", result)

        return result

    
    def get_i(self):
        """
        Get intentions
        """
        targets_prop = self.get_prop_intentions()
        targets_vis = self.get_vis_intentions()
        targets_needs = np.tile(self.mu[0,:c.needs_len],(c.num_intentions,1))
        targets_focus = self.get_focus_intentions()

        targets = np.concatenate((targets_needs,targets_prop,targets_vis,targets_focus),axis=1) # concatenate to get final matrix of shape NUM_INTENTIONS x (NEEDS_LEN + PROP_LEN + LATENT_SIZE + FOCUS_LEN)

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
        self.E_i = (I - self.mu[0]) * c.k

        return self.mu[1] - self.E_i
    
    def get_sensory_precisions(self, S):
        
        pi_vis, dPi_dmu0_vis, dPi_dmu1_vis = utils.pi_foveate(np.ones((c.height,c.width)), self.mu[0])

        dim = c.needs_len+c.prop_len+c.latent_size+c.focus_len

        Pi = [np.ones(dim) * c.pi_need,
              np.ones(dim) * c.pi_prop, 
              pi_vis]
        
        dPi_dmu0 = [np.zeros((dim,dim)), 
                    np.zeros((dim,dim)), 
                    dPi_dmu0_vis]
        
        dPi_dmu1 = [np.zeros((dim,dim)), 
                    np.zeros((dim,dim)), 
                    dPi_dmu1_vis]

        return Pi, dPi_dmu0, dPi_dmu1
    
    def get_intention_precisions(self, S):
        sm  = softmax(np.array([self.mu[0,2],self.mu[0,5]])).reshape(2, 1)
        self.beta = sm * self.beta_weights

        dGamma_dmu0 = [np.zeros((self.belief_dim, self.belief_dim))] * c.num_intentions 

        dGamma_dmu1 = [np.zeros((self.belief_dim, self.belief_dim))] * c.num_intentions

        return self.beta, dGamma_dmu0, dGamma_dmu1
    
    
    def get_likelihood(self, E_s, grad_v, Pi):
        """
        Get likelihood components
        """
        lkh = {}
        lkh['need'] = self.alpha[0] * Pi[0] * E_s[0].dot(self.G_n.T)

        lkh['prop'] = self.alpha[1] * Pi[1] * E_s[1].dot(self.G_p.T)

        lkh['vis'] = self.alpha[2] * self.vae.get_grad(*grad_v, torch.from_numpy(Pi[2])*E_s[2])
        lkh['vis'] = np.concatenate((np.zeros((c.needs_len+c.prop_len)),lkh['vis'],np.zeros(c.focus_len))) 

        return lkh


    def attention(self, precision, derivative, error):
        total = np.zeros(self.belief_dim)
        for i in range(len(precision)):
            component1 = 0.5 * np.mean(np.expand_dims(1/precision[i], axis=-1) * derivative[i], axis=tuple(range(derivative[i].ndim - 1)))
            component1[-3] = 0.1 * component1[-3]
            component1[-2:] = c.attn_damper1 * component1[-2:]
            component2 = -0.5 * np.sum(np.expand_dims(error[i]**2, axis=-1) * derivative[i], axis=tuple(range(derivative[i].ndim - 1)))
            component2[-3] = c.attn_damper2 * component2[-3]

            if i==2:
                print("c1", component1)
                print("c2", component2)
            total += component1 + component2

        return total

    def get_mu_dot(self, lkh, E_s, E_mu, Pi, Gamma, dPi_dmu0, dGamma_dmu0, dPi_dmu1, dGamma_dmu1):
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

        generative = lkh['prop'] + lkh['need'] + lkh['vis']
        backward = - c.k * forward_i

        bottom_up0 = self.attention(Pi,dPi_dmu0,e_s)
        top_down0 = self.attention(Gamma,dGamma_dmu0,E_mu)

        bottom_up1 = self.attention(Pi, dPi_dmu1,[0]*3) # No sensory error for second order
        top_down1 = self.attention(Gamma,dGamma_dmu1,[0]*c.num_intentions) # No intention error for second order

        # print("\nmu_dot[0]>")
        # print("self.mu[1]", self.mu[1], np.linalg.norm(self.mu[1]))
        print("generative", generative, np.linalg.norm(generative))
        # print("backward", backward[4:6], np.linalg.norm(backward))
        print("bottom_up0", bottom_up0, np.linalg.norm(bottom_up0))
        # print("top_down0", top_down0)

        # print("\nmu_dot[1]>")
        # print("-forward_i", -forward_i[4:6], np.linalg.norm(-forward_i))
        # print("bottom_up1",bottom_up1)
        # print("top_down1", top_down1)

        self.mu_dot[0] = self.mu[1] + generative + backward + bottom_up0 + top_down0 #
        self.mu_dot[1] = -forward_i + bottom_up1 + top_down1
        print("mu_dot0 before clip:",self.mu_dot[0], np.linalg.norm(self.mu_dot[0]))
        self.mu_dot = np.clip(self.mu_dot,-0.25,0.25) # clip mu update


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
        # print("a_dot",self.a_dot)

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
        self.mu = np.clip(self.mu,-1,1) # clip mu values
        self.mu[:,c.needs_len+c.prop_len+c.latent_size] = np.clip(self.mu[:,c.needs_len+c.prop_len+c.latent_size],c.pi_vis,1) # clip mu_amp
        print("self.mu[0]",self.mu[0])
        self.vectors[2,:] = self.mu[0,-2:]

        # Update action
        self.a += c.dt * self.a_dot
        self.a = np.clip(self.a, -c.a_max, c.a_max)

    def init_belief(self, needs, prop, visual):
        """
        Initialize belief
        """
        visual_state =self.vae.predict_latent(visual.squeeze()).detach().squeeze().numpy()
        focus = np.array([c.pi_vis,0,0])

        # package_share_directory = get_package_share_directory(c.package_name)
        # for f_name in [os.path.join(package_share_directory, 'resource', x) for x in c.focus_samples]:
        #     focus_samples = np.loadtxt(f_name,delimiter=",",dtype="float32")
        #     self.focus_samples.append(focus_samples)
        self.mu[0] = np.concatenate((needs, prop, visual_state,focus)) # initialize with beliefs about needs, proprioceptive and visual state
        print("mu initialized to:",self.mu[0])

        sm = softmax(np.array([self.mu[0,2],self.mu[0,5]])).reshape(2, 1)
        self.beta = sm * self.beta_weights


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
        
        # print("mu:",self.mu[0])
        
        # Get predictions
        P, grad_v = self.get_p()

        # Get intentions
        I = self.get_i()

        # Get sensory prediction errors
        E_s = self.get_e_s(S, P)

        # Get dynamics prediction errors
        E_mu = self.get_e_mu(I)

        # Get sensory precisions
        Pi, dPi_dmu0, dPi_dmu1 = self.get_sensory_precisions(S)

        # Get intention precisions
        Gamma, dGamma_dmu0, dGamma_dmu1 = self.get_intention_precisions(S)

        # Get likelihood components
        likelihood = self.get_likelihood(E_s, grad_v, Pi)

        # Get belief update
        self.get_mu_dot(likelihood, E_s, E_mu, Pi, Gamma, dPi_dmu0, dGamma_dmu0, dPi_dmu1, dGamma_dmu1)

        # Get action update
        self.get_a_dot(likelihood, Pi) # E_s[0] * self.pi_s[0] * self.alpha[0]

        # Update
        self.integrate()

        # Show visual sensory and predicted data
        utils.show_SP(S, P, self.vectors)

        # Start action
        self.switch_mode(step)

        return self.a, np.linalg.norm(likelihood["vis"]), np.linalg.norm(E_s[2])
