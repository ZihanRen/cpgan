#%%
from model import Generator,Discriminator 
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

# I/0
f_yaml = init_yaml.yaml_f
img_path = f_yaml['img_path']['img_chunk']
feature_path = path.join(f_yaml['feature_path'],'features.csv')
df = pd.read_csv(feature_path)
names_img = df['name'].to_list()

experiment_index = 1
torch.cuda.empty_cache()
batch_size = 25
lr = 7e-04
z_dim = 100
n_epochs = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

print('Running device available: {}'.format(torch.cuda.get_device_name(0)))


random.shuffle(names_img)
names_img = names_img[0:9000]

dataGAN = Dataset3D(img_path,feature_path,names_img)
train_data_loader = DataLoader(dataGAN,batch_size=batch_size,shuffle=True,drop_last=True)

criterion = nn.BCEWithLogitsLoss()
gen = Generator(f_dim=1,embed_size=6).to(device)
disc = Discriminator().to(device)
optimizerD = optim.Adam(disc.parameters(),lr=lr)
optimizerG = optim.Adam(gen.parameters(),lr=lr)

# %%
img_list = []
G_losses = []
D_losses = []
torch.backends.cudnn.benchmark = False
# beginning training
print('Training begin: ...')

for epoch in range(n_epochs):
  start = time.time()
  
  for i, (real_images,features) in enumerate(train_data_loader):
    real_images = real_images.to(device)
    features = features.to(device)

    b_size = len(real_images)
    noise = torch.randn(b_size,z_dim,device=device)

    ## train discriminator ##
    optimizerD.zero_grad()

    # train real vs real
    pred_real = disc(real_images,features)
    d_loss_real = criterion(pred_real,torch.ones_like(pred_real))
    D_x = pred_real.mean().item()

    # train fake vs fake #
    fake_images = gen(noise,features)
    pred_fake = disc(fake_images,features)
    d_loss_fake = criterion(pred_fake,torch.zeros_like(pred_fake))
    d_loss_all = (d_loss_fake + d_loss_real) / 2
    
    d_loss_all.backward()
    optimizerD.step()
    D_G_z1 = pred_fake.mean().item()


    ## train generator ## 
    # preprocess data set to embeded features
    noise2 = torch.randn(b_size,z_dim,device=device)

    # begin training
    optimizerG.zero_grad()
    fake_images_2 = gen(noise2,features)
    fake_pred = disc(fake_images_2,features)
    g_loss = criterion(fake_pred,torch.ones_like(fake_pred))
    g_loss.backward()
    optimizerG.step()
    D_G_z2 = fake_pred.mean().item()

  ## Output training stats ##
  print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        % (epoch, n_epochs, i, len(train_data_loader),
            d_loss_all.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
  
  G_losses.append(g_loss.item())
  D_losses.append(d_loss_all.item())
  # save model after each epoch
  if epoch%10 == 0:
    save_path = f'train_hist/cgan06{epoch}.pth'
    torch.save(gen.state_dict(),  save_path)
    f = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    f.savefig('train_hist/cgan_profile{}.png'.format(epoch))
  print("Training time takes approximately: ", (time.time() - start)/60, "minutes at current epcoh")




# %%
