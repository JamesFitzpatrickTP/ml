import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import torch

import layers

img_path = '/mnt/c/Users/James Fitzpatrick/Documents/P00200/00_original'
lab_path = '/mnt/c/Users/James Fitzpatrick/Documents/P00200/05_hollow'
imgs = os.listdir(img_path)
labs = os.listdir(lab_path)
imgs = np.stack([pydicom.read_file(os.path.join(img_path, img)).pixel_array for img in imgs])
labs = np.stack([pydicom.read_file(os.path.join(lab_path, lab)).pixel_array for lab in labs])
labs = (labs > 0.1).astype(float)
                
model = layers.SmallNetwork()
# optimiser = torch.optim.Adam(model.parameters(), lr=0.001,
#                              betas=(0.9, 0.999), amsgrad=True)
optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()


def to_tensor(vol, idx, expand=True):
    ten = vol[idx][::8,::8]
    if expand:
        ten = np.expand_dims(ten, 0)
        ten = np.expand_dims(ten, 0)
    return torch.tensor(ten).float()
    

for epoch in range(200):
    print('###### EPOCH {} ######'.format(epoch))
    for idx in range(150, 151):
        print(idx)
        img = to_tensor(imgs, idx, expand=True)
        lab = to_tensor(labs, idx, expand=False)
        pred = model(img)
        loss = criterion(pred, lab)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        plt.imsave('/mnt/c/Users/James Fitzpatrick/Downloads/imgs/' + str(epoch), pred.data.numpy())
        print("##### ERROR: ", loss, ' ######')
        
