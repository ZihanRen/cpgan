#%%
import glob
import re
import imageio

images = []

# Find all PNG files in the current directory with the naming 
# pattern and add them to the list
for filename in sorted(
    glob.glob('img_phi_*.png'), 
    key=lambda x: int(re.findall('\d+', x)[0])
    ):
    images.append(imageio.imread(filename))

# Set the duration of all frames after frame 30 to 10 seconds
for i in range(31, len(images)):
    images[i] = images[i].astype('uint8')
    images[i] *= 0  # set all pixels to black
    images[i][0, 0] = 255  # set top-left pixel to white
    images[i] = images[i].astype('uint8')
    
# Save the GIF with a loop of 1
gif_path = 'output.gif'  # replace with the name of the output GIF file
imageio.mimsave(gif_path, images, duration=0.2, loop=10)
# %%
