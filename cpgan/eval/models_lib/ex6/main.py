#%%
from cpgan.train import generator,discriminator
from cpgan.train import utils as utl
from cpgan.train.dataset import Dataset3D
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import random
import time
from cpgan import init_yaml
import pandas as pd
from os import path
import os
import pandas as pd
from torchvision.utils import make_grid
from torchvision.utils import save_image

def img_save(img_fake,PATH,sample_num=4):
    img_fake = img_fake[:,:,30,::]
    img_fake = img_fake.detach().cpu()
    image_grid = make_grid(img_fake[:sample_num],nrow=2)
    plt.imshow(image_grid.permute(1, 2, 0))
    save_image(image_grid,PATH)

# check hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
print('Running device available: {}'.format(torch.cuda.get_device_name(0)))
ex = 6

# I/0 - shuffle names and images
f_yaml = init_yaml.yaml_f
img_path = f_yaml['img_path']['img_chunk']
trainhist_path = f_yaml['model']
# create saving space
os.makedirs(path.join(trainhist_path,f'ex{ex}'),exist_ok=True)
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
display_step = 500
dataGAN = Dataset3D(img_path,feature_path,names_img)
train_data_loader = DataLoader(
  dataGAN,batch_size=batch_size,shuffle=True,drop_last=True
  )

g_lambda = 10 # weight of gradident penalty

d_lr = 1e-4
g_lr = 5e-4
crit_repeat = 5
gen = generator.Generator(z_dim=z_dim).to(device)
disc = discriminator.Discriminator().to(device)
optimizerD = optim.Adam(disc.parameters(),lr=d_lr)
optimizerG = optim.Adam(gen.parameters(),lr=g_lr)

# fixed noise injection - for checking image quality
noise_fix = torch.randn(10,200).to(device)

#%%
# monitor the losses
img_list = []
G_losses = []
D_losses = []
penalty_loss = []

torch.backends.cudnn.benchmark = False
# beginning training
print('Training begin: ...')

train_step = 0

for epoch in range(n_epochs):
  start = time.time()
  

  for i, (real_images,phi) in enumerate(train_data_loader):
    real_images = real_images.to(device)
    noise = torch.randn(batch_size,z_dim,device=device)
    
    mean_iter_crit_loss = 0
    mean_iter_gp_loss = 0
    ####### train critic #########
    for _ in range(crit_repeat):
      optimizerD.zero_grad()
      pred_real = disc(real_images)
      fake_images = gen(noise)
      pred_fake = disc(fake_images.detach())
      gp = utl.gradident_penalty(disc,real_images,fake_images,device=device)
      d_loss = -(torch.mean(pred_real) - torch.mean(pred_fake)) + g_lambda*gp
      d_loss.backward(retain_graph=True)
      optimizerD.step()
      mean_iter_crit_loss += d_loss.item() / crit_repeat
      mean_iter_gp_loss += gp.item() / crit_repeat

    ########### train generator ##############
    optimizerG.zero_grad()
    pred_fake = disc(fake_images)
    g_loss = -torch.mean(pred_fake)
    g_loss.backward()
    optimizerG.step()
    
    ############# track the losses ###############
    D_losses.append(mean_iter_crit_loss)
    penalty_loss.append(mean_iter_gp_loss)
    G_losses.append(g_loss.item())

    ####### Display training progress #############
    if train_step%display_step == 0 and train_step > 0:
      D_loss_mean = sum(D_losses[-display_step:]) / display_step
      G_loss_mean = sum(G_losses[-display_step:]) / display_step      
      penalty_loss_mean = sum(penalty_loss[-display_step:]) / display_step


      img_save_PATH = path.join(trainhist_path,f'ex{ex}',f'img{ex}-{train_step}.png')
      img_fake = gen(noise_fix)
      img_save(img_fake,PATH=img_save_PATH)
      
      print(f"\nEpoch {epoch}, step {train_step} result:")
      print(f"Generator loss: {G_loss_mean}")
      print(f"Discriminator loss: {D_loss_mean}")
      print(f"Penalty loss: {penalty_loss_mean}")

    train_step += 1

  # save model after each epoch
  if epoch%5 == 0:
    models_save = path.join(trainhist_path,f'ex{ex}',f'cganex{ex}-{epoch}.pth')
    torch.save(gen.state_dict(),models_save)
    f = plt.figure(figsize=(10,5))
    plt.title(f"Generator and Discriminator Loss During Training - Epoch {epoch}")
    plt.plot(G_losses,c='r',label="G-Loss")
    plt.plot(D_losses,c='b',label="D-Loss")
    plt.plot(penalty_loss,c='g',label="Penalty loss")

    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    profile_save = path.join(trainhist_path,f'ex{ex}','cgan_profile{}.png'.format(epoch))
    f.savefig(profile_save)
    
    loss_all = {
    'G-loss':G_losses,
    'D-loss':D_losses,
    "P-loss":penalty_loss
    } 

    df = pd.DataFrame(loss_all)
    columns=['G_loss','D_loss',"penalty_loss"]
    df.to_csv(path.join(trainhist_path,f'ex{ex}',f'loss_history_{epoch}.csv'),index=False)

  print("Training time takes approximately: ", (time.time() - start)/60, "minutes at current epcoh")



# %%
