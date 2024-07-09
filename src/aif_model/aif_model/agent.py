import aif_model.vae as vae
import aif_model.config as c
import aif_model.networks as networks
import aif_model.utils as utils
import torch
import numpy as np
import cv2

class Agent:
    """
    Active Inference agent
    """
    def __init__(self):

        # Load networks
        vae_path = c.vae_path
        intentions_path = c.intentions_path

        self.vectors = np.zeros((2,2))

        self.vae  = vae.VAE( # VAE network
        latent_dim=c.latent_size,
        encoder=networks.Encoder(in_chan=c.channels, latent_dim=c.latent_size),
        decoder=networks.Decoder(out_chan=c.channels, latent_dim=c.latent_size))
        self.vae.load(vae_path)

        # self.int_net = networks.FullyConnected(c.latent_size, c.prop_len * c.num_intentions) # intention network
        # self.int_net.load(intentions_path)

        belief_dim = c.needs_len + c.prop_len + c.latent_size #+ c.latent_size*c.num_intentions # needs, proprioceptive belief, visual belief, visual intentions

        # Initialization of variables
        self.mu = np.zeros((c.n_orders, belief_dim), dtype="float32") 
        self.mu_dot = np.zeros_like(self.mu)
        self.focus_samples = []

        self.a = np.zeros(c.prop_len)
        self.a_dot = np.zeros_like(self.a)

        self.E_i = np.zeros((c.num_intentions, belief_dim))

        self.alpha = np.array([1, c.alpha, 1-c.alpha]) # needs, proprioceptive, visual
        self.pi_s = np.array([c.pi_need,c.pi_prop,c.pi_vis])

        self.beta_index = 0
        weights = [] #np.array([[1]*c.needs_len+[1]*c.prop_len+[0.7e-1]*c.latent_size*(c.num_intentions+1)])
        for i in range(c.num_intentions):
            builder = np.array([[1]*c.needs_len+[1]*c.prop_len+[1e-1]*c.latent_size])#np.array([[1]*c.needs_len + [1]*c.prop_len + [0.0]*(i*c.prop_len) +[0.7e-1]*c.prop_len + [0.0]*((c.num_intentions-1-i)*c.prop_len + c.latent_size-c.prop_len*c.num_intentions) + [0.0]*c.latent_size*c.num_intentions])
            weights.append(builder)

        self.beta_weights = np.array(weights)
        #self.beta = np.zeros(c.num_intentions)*1e-4; self.beta[0] = 1e-2
        self.mode = "closest"

        # Generative models (simple)
        self.G_p = utils.shift_rows(np.eye(belief_dim, c.prop_len),c.needs_len)

        self.G_n = np.eye(belief_dim, c.needs_len)

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
                print("Focus sample id for object "+str(i)+" is "+str(closest)+" , diff is",diff[closest])
            elif self.mode == "mean":
                targets_vis.append(np.mean(self.focus_samples[i],axis=0))

        return np.array(targets_vis)
    
    def get_i(self):
        """
        Get intentions
        """
        # targets = self.int_net(torch.tensor(self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.latent_size])).detach().numpy() # pass visual belief into intention conversion net
        targets = self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.prop_len*c.num_intentions] # grab visual positions of objects
        # targets = np.expand_dims(targets, axis=0)
        targets = np.reshape(targets,(c.num_intentions,c.prop_len)) # reshape
        targets = utils.denormalize(targets) # convert from range [-1,1] to [0,width]
        print("Target in pixels:",targets)
        self.vectors = np.array(targets)
        targets = utils.pixels_to_angles(targets) # convert to angles
        # print("Targets in relative angles:",targets)

        targets_prop = targets + self.mu[0,c.needs_len:c.needs_len+c.prop_len] # add relative target angle to global camera angle
        targets_vis = self.get_vis_intentions()
        # targets_vis = self.mu[0,c.needs_len+c.prop_len+c.latent_size:] # intention focus representations
        # targets_vis = np.reshape(targets_vis,(c.num_intentions,c.latent_size))
        # targets_vis = np.tile(self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.latent_size],(c.num_intentions,1)) # copy current visual belief
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
    
    def get_likelihood(self, E_s, grad_v):
        """
        Get likelihood components
        """
        lkh = {}
        lkh['need'] = self.alpha[0] * self.pi_s[0] * E_s[0].dot(self.G_n.T)

        lkh['prop'] = self.alpha[1] * self.pi_s[1] * E_s[1].dot(self.G_p.T)

        lkh['vis'] = self.alpha[2] * self.pi_s[2] * self.vae.get_grad(*grad_v, E_s[2])
        lkh['vis'] = np.concatenate((np.zeros((c.needs_len+c.prop_len)),lkh['vis'])) #np.concatenate((np.zeros((c.needs_len+c.prop_len)),lkh['vis'],np.zeros((c.num_intentions*c.latent_size))))

        # print("E_s[1]",E_s[1])
        # print("lkh_prop",lkh["prop"])
        # print("lkh_vis",lkh["vis"][c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.latent_size])
        # lkh_pix = (lkh["vis"][2:5]+1)/2*c.width
        # print("lkh_vis in pixels",lkh_pix)
        # lkh_angles = utils.pixels_to_angles(np.expand_dims(lkh_pix,axis=0))
        # print("lkh_vis in angles",lkh_angles)

        return lkh
    
    def get_mu_dot(self, lkh, E_mu):
        """
        Get belief update
        """
        
        self.mu_dot = np.zeros_like(self.mu)

        # Intention components
        forward_i = np.zeros((c.needs_len + c.prop_len + c.latent_size)) #np.zeros((c.needs_len + c.prop_len + c.latent_size + c.num_intentions*c.latent_size))
        for g, e in zip(self.beta, np.array(E_mu)):
            forward_i += g * e

        self.mu_dot[0] = self.mu[1] + lkh['prop'] + lkh['need'] + lkh['vis']
        self.mu_dot[1] = -forward_i
        # print("mu_dot:",self.mu_dot)

    def get_a_dot(self, likelihood):
        """
        Get action update
        """
        e_prop = likelihood["prop"].dot(self.G_p)
        # print("e_prop",e_prop)

        old_lkh_vis= np.array(likelihood["vis"][c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.prop_len*c.num_intentions], dtype="float32")
        old_lkh_vis = np.reshape(old_lkh_vis,(c.num_intentions,c.prop_len))
        old_lkh_pix = utils.denormalize(old_lkh_vis)
        old_lkh_ang = utils.pixels_to_angles(old_lkh_pix)

        # lkh_vis = np.array(likelihood["vis"][c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.latent_size], dtype="float32")
        # lkh_vis = self.int_net(torch.tensor(lkh_vis)).detach().numpy()
        # lkh_vis = np.reshape(lkh_vis,(c.num_intentions,c.prop_len))
        # lkh_pix = utils.denormalize(lkh_vis)
        # lkh_angles = utils.pixels_to_angles(lkh_pix)

        # index = np.argmax(np.linalg.norm(lkh_angles,axis=1))# max = biggest surprise; 
        index = self.beta_index#np.argmax(self.beta) # from beta = most desired; 
        
        old_lkh_angles = old_lkh_ang[index]
        # lkh_angles = lkh_angles[index]
        # old_lkh_angles = np.mean(old_lkh_ang,axis=0)# avg = average movement

        print("old_lkh_ang:",old_lkh_angles)
        # print("new_lkh_ang:",lkh_angles)

        d_mu_lkh_vis = c.dt * old_lkh_angles
        d_mu_lkh_prop = -c.dt * e_prop

        print("dmu_lkh_vis",d_mu_lkh_vis)
        print("dmu_lkh_prop",d_mu_lkh_prop)

        self.a_dot = d_mu_lkh_vis + d_mu_lkh_prop #c.alpha * d_mu_lkh_prop + (1 - c.alpha) * d_mu_lkh_vis
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
        # TODO: try clipping self.mu
        # self.mu = np.clip(self.mu,-1,1)

        # Update action
        self.a += c.dt * self.a_dot
        self.a = np.clip(self.a, -c.a_max, c.a_max)

    def init_belief(self, needs, prop, visual):
        """
        Initialize belief
        """
        visual_state =self.vae.predict_latent(visual.squeeze()).detach().squeeze().numpy()
        for f_name in c.focus_samples:
            focus_samples = np.loadtxt(f_name,delimiter=",",dtype="float32")
            self.focus_samples.append(focus_samples)
        self.mu[0] = np.concatenate((needs, prop, visual_state)) # initialize with beliefs about needs, proprioceptive and visual state
        print("mu initialized to:",self.mu[0])

        self.beta_index = np.argmax(needs)
        self.beta = np.zeros((c.num_intentions,c.needs_len+c.prop_len+c.latent_size)); self.beta[self.beta_index] = self.beta_weights[self.beta_index]


    def switch_mode(self, step):
        print("mode",self.mode)
        if step%5== 0:
            # self.mode="mean"
            if self.mode=="closest": self.mode = "mean"
            elif self.mode=="mean": self.mode = "closest"
        # else:
        #     self.mode = "closest"

    def attention(self, step):
        self.beta_index = np.argmax(self.mu[0,:c.needs_len])
        self.beta = np.zeros((c.num_intentions,c.needs_len+c.prop_len+c.latent_size)); self.beta[self.beta_index] = self.beta_weights[self.beta_index]
        print("beta index:",self.beta_index)
        # timer = 300#10*c.update_frequency
        # if step%timer == 0:
        #     self.beta = utils.shift_rows(self.beta,1)
        #     self.beta_index+=1
        #     self.beta_index%=c.num_intentions
        #     print("--------------------------------------------Attention shift!")
        # Attention decay
        # if step%3 == 0:
        #     self.beta *= 0.995

    def inference_step(self, S, step):
        """
        Run an inference step
        """

        print("mu[0]:",self.mu[0])
        if (self.mu == np.nan).any():
            print("nan in mu",self.mu)
            print("S",S)
            raise Exception("nan in mu")
        # print("visual encodings denormalized:")
        # print(utils.denormalize(self.mu[0,c.needs_len+c.prop_len:]))
        # Get predictions
        P, grad_v = self.get_p()
        # cv2.imshow("predicted image",cv2.cvtColor(np.transpose(P[2],(1,2,0)),cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        # recon, _, _ = self.vae.forward(S[2])
        # recon = np.transpose(recon[0].detach().squeeze().numpy(),(1,2,0))
        # tmp = np.transpose(S[2].detach().squeeze().numpy(),(1,2,0))
        # cv2.imshow("reconstruction test",cv2.cvtColor(np.concatenate((tmp,recon,np.transpose(P[2],(1,2,0))),axis=0),cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Get intentions
        I = self.get_i()

        # Get attention weights

        self.attention(step)

        # Get sensory prediction errors
        E_s = self.get_e_s(S, P)

        # Get dynamics prediction errors
        E_mu = self.get_e_mu(I)
        # print("I",I)
        # print("E_i",self.E_i)
        # print("mu[1]",self.mu[1])
        # print("E_mu",E_mu)

        # Get likelihood components
        likelihood = self.get_likelihood(E_s, grad_v)
        # print("E_s",E_s)
        # print("lkh",likelihood)

        # Get belief update
        self.get_mu_dot(likelihood, E_mu)

        # Get action update
        self.get_a_dot(likelihood) # E_s[0] * self.pi_s[0] * self.alpha[0]

        # Update
        self.integrate()

        # _, new_prediction = self.vae.predict_visual(torch.tensor(self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.latent_size]).unsqueeze(0))
        f = 5
        self.tmp_S = np.transpose(S[2].detach().squeeze().numpy(),(1,2,0))
        self.tmp_S = cv2.resize(self.tmp_S,(0,0),fx=f,fy=f)
        # self.tmp_S = utils.display_vectors(self.tmp_S,(self.vectors-16)/32)
        self.tmp_P = np.transpose(P[2],(1,2,0))
        self.tmp_P = cv2.resize(self.tmp_P,(0,0),fx=f,fy=f)
        self.tmp_P = utils.display_vectors(self.tmp_P,(self.vectors-16)/32)
        combined = np.concatenate((self.tmp_S,self.tmp_P), axis=1) # + ,np.transpose(new_prediction.detach().squeeze().numpy(),(1,2,0))
        combined = cv2.cvtColor(combined,cv2.COLOR_RGB2BGR)
        cv2.imshow("S,P",combined)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()

        # Start action
        self.switch_mode(step)

        return self.a
