#%%
from cpgan.train import generator,discriminator
from torch import nn
from Dataset3D import Dataset3D
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import random
import time
from cpgan import init_yaml
import pandas as pd
from os import path
import pandas as pd

def combine_vectors(x, y):
    combined = torch.cat([x.float(), y.float()], 1)
    return combined

def tensor_process(image_tensor):

    image_unflat = image_tensor.detach().cpu()
    image_numpy = image_unflat.numpy()

    return image_numpy

def img_filter(im):
  return im>0.5

def control_vec(phi,other_c,sample_num):
    phi_arr = torch.ones(sample_num,1).to(device)*phi    
    combine_c = combine_vectors(phi_arr,other_c)
    return combine_c

def img_save(PATH):
    with torch.no_grad():
      sample_num = 2
      other_c = torch.randn(sample_num,1).to(device)
      c_i = control_vec(0.2,other_c,sample_num)
      noise_eval = torch.randn(2,200).to(device)
      z_test = combine_vectors(c_i,noise_eval)
      img_fake = gen(z_test)
      img_fake = tensor_process(img_fake)
      img_fake = img_filter(img_fake)
      f = plt.figure()
      plt.imshow(img_fake[0,0,0,:,:],cmap='binary')
      f.savefig(PATH)

def get_gradient(crit, real, fake, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)[0]
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean( (gradient_norm-1)**2 )
    return penalty

def get_gen_loss(w_loss,info_loss,i_lambda):
    gen_loss = w_loss + info_loss*i_lambda 
    return gen_loss

def get_crit_loss(w_loss, grad_penalty_loss,g_lambda,info_loss,i_lambda):
    crit_loss = w_loss + grad_penalty_loss*g_lambda + info_loss*i_lambda
    return crit_loss

# check hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
print('Running device available: {}'.format(torch.cuda.get_device_name(0)))
ex = 5

# I/0 - shuffle names and images
f_yaml = init_yaml.yaml_f
img_path = f_yaml['img_path']['img_chunk']
trainhist_path = f_yaml['model']
feature_path = path.join(f_yaml['feature_path'],'features.csv')
df = pd.read_csv(feature_path)
names_img = df['name'].to_list()
random.seed(2022)
random.shuffle(names_img)
names_img = names_img[0:15000]

# initial parameters
batch_size = 25
z_dim = 200
n_epochs = 100
c_dim = 2 # 1 porosity, the rest properties will be inferred from GAN
display_step = 100
dataGAN = Dataset3D(img_path,feature_path,names_img)
train_data_loader = DataLoader(
  dataGAN,batch_size=batch_size,shuffle=True,drop_last=True
  )

adv_loss = nn.BCEWithLogitsLoss()
i_loss = nn.MSELoss(reduction='none')

i_lambda = 0.9 # weight of info loss
g_lambda = 10 # weight of gradident penalty

d_lr = 2e-4
g_lr = 1e-3
gen = generator.Generator(z_dim=z_dim+c_dim).to(device)
disc = discriminator.Discriminator(c_dim=c_dim).to(device)
optimizerD = optim.Adam(disc.parameters(),lr=d_lr)
optimizerG = optim.Adam(gen.parameters(),lr=g_lr)


# %%
# monitor the losses
img_list = []
G_losses = []
D_losses = []
I_losses = []
penalty_loss = []
phi_losses = []

torch.backends.cudnn.benchmark = False
# beginning training
print('Training begin: ...')

train_step = 0

for epoch in range(n_epochs):
  start = time.time()
  
  for i, (real_images,phi) in enumerate(train_data_loader):
    real_images = real_images.to(device)

    # gaussain space programming: supervised + unsupervised + noise
    c_phi = phi.to(device).float()
    c_other = torch.randn(batch_size,1,device=device)
    c_input = combine_vectors(c_phi,c_other)
    noise = torch.randn(batch_size,z_dim,device=device)
    combine_input = combine_vectors(c_input,noise)
    
    ####### train critic #########
    optimizerD.zero_grad()
    # adversarial loss for discriminator
    pred_real,_ = disc(real_images)
    fake_images = gen(combine_input)
    pred_fake,pred_c = disc(fake_images.detach())
    d_loss_adv = -torch.mean(pred_real) + torch.mean(pred_fake)
    # info loss
    info_loss = torch.sum(i_loss(c_input,pred_c),dim=0)
    # extract porosity mutual info from loss vector
    phi_losses.append(info_loss[0].item())
    i_loss_mean = info_loss.mean()
    # penalty term
    epsilon = torch.rand(len(real_images),1,1, 1, 1, device=device, requires_grad=True)
    gradient = get_gradient(disc, real_images, fake_images.detach(), epsilon)
    gp = gradient_penalty(gradient)
    # aggregate all loss together
    d_loss_all = get_crit_loss(d_loss_adv,gp,g_lambda,i_loss_mean,i_lambda)
    d_loss_all.backward(retain_graph=True)
    optimizerD.step()
    # track the losses
    D_losses.append(d_loss_adv.item())
    I_losses.append(i_loss_mean.item())
    penalty_loss.append(gp.item())
    
    ########### train generator ##############
    # adv loss
    optimizerG.zero_grad()
    pred_fake,pred_c = disc(fake_images)
    g_loss_adv = -torch.mean(pred_fake)
    # Info-Loss is the same as previuos one, but in a new computational graph
    info_loss= i_loss(c_input,pred_c)
    info_loss_mean = torch.sum(info_loss,dim=0).mean()

    g_loss_all = get_gen_loss(g_loss_adv,info_loss_mean,i_lambda)
    g_loss_all.backward()
    optimizerG.step()
    # track the losses
    G_losses.append(g_loss_adv.item())

    if train_step%display_step == 0 and train_step > 0:
      D_loss_mean = sum(D_losses[-display_step:]) / display_step
      G_loss_mean = sum(G_losses[-display_step:]) / display_step      
      I_loss_mean = sum(I_losses[-display_step:]) / display_step
      phi_loss_mean = sum(phi_losses[-display_step:]) / display_step
      penalty_loss_mean = sum(penalty_loss[-display_step:]) / display_step


      img_save_PATH = path.join(trainhist_path,f'ex{ex}',f'img{ex}-{train_step}.png')
      img_save(img_save_PATH)
      
      print(f"\nEpoch {epoch}, step {train_step} result:")
      print(f"Generator loss: {G_loss_mean}")
      print(f"Discriminator loss: {D_loss_mean}")
      print(f"Mutual Information: {I_loss_mean}")
      print(f"Mutual Porosity: {phi_loss_mean}")
    train_step += 1

  # save model after each epoch
  if epoch%5 == 0:
    models_save = path.join(trainhist_path,f'ex{ex}',f'cganex{ex}-{epoch}.pth')
    torch.save(gen.state_dict(),models_save)
    f = plt.figure(figsize=(10,5))
    plt.title(f"Generator and Discriminator Loss During Training - Epoch {epoch}")
    plt.plot(G_losses,c='r',label="G-Loss")
    plt.plot(D_losses,c='b',label="D-Loss")
    plt.plot(I_losses,c='y',label="I-mutual")
    plt.plot(phi_losses,c='g',label="Porosity mutual")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    profile_save = path.join(trainhist_path,f'ex{ex}','cgan_profile{}.png'.format(epoch))
    f.savefig(profile_save)

  print("Training time takes approximately: ", (time.time() - start)/60, "minutes at current epcoh")

loss_all = {
  'G-loss':G_losses,
  'D-loss':D_losses,
  'I-loss':I_losses,
  'phi-loss':phi_losses
  }

df = pd.DataFrame(loss_all)
columns=['G_loss,D_loss,I_loss,phi_loss']
df.to_csv('loss_history.csv',index=False)

# %%
