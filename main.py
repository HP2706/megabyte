from PIL import Image
import torch
import numpy as np
from utils import img_to_bytes, bytes_to_img, text_to_bytes, bytes_to_text, MnistDataloader, patch_images
from model import Megabyte, Megabyte_Config, Patch_Embedder, VisualTransformer, VisualTransformer_Config, ResNet, ResNet_Config
from torch.utils.data import DataLoader
from functools import partial
import tqdm

#model = VisualTransformer(VisualTransformer_Config())

def vit_collate_fn(batch, vit=False):
    images, labels = zip(*batch) #(Tensor, Tensor)
    images = torch.from_numpy(np.stack(images)).to(torch.float32)
    labels = torch.from_numpy(np.stack(labels)).to(torch.int64)
    print("labels.shape", labels.shape)
    images = patch_images(images)
    return images, labels

def convnet_collate_fn(batch, convnet=False):
    images, labels = zip(*batch) #(Tensor, Tensor)
    images = torch.from_numpy(np.stack(images)).to(torch.float32)
    labels = torch.from_numpy(np.stack(labels)).to(torch.int64)
    #add channel_dim
    images = images.unsqueeze(1)
    return images, labels

#images = patch_images(images) if Vit

def megabyte_collate_fn(batch, type="image"):
    if type == "image":
        images, labels = zip(*batch)
        images = np.stack(images) # we need it to be a numpy array to use patch_images
        bytes = img_to_bytes(images) # should be integer type
        labels = torch.Tensor(labels).to(torch.int64)
        #print("bytes.shape", bytes.shape)
    elif type == "text":
        texts, labels = zip(*batch)
        texts = torch.from_numpy(np.array(texts))
        bytes = torch.Tensor(text_to_bytes(texts))
    return bytes, labels 


model = ResNet(ResNet_Config())   #Megabyte(Megabyte_Config())
(x_train, y_train), (x_test, y_test) = MnistDataloader().load_data()


print("vit params", VisualTransformer(VisualTransformer_Config()).get_param_count())
print("megabyte params", Megabyte(Megabyte_Config()).get_param_count())
print("convnet params", model.get_param_count())


train_dataloader = DataLoader(list(zip(x_train, y_train)), batch_size=128, shuffle=True, collate_fn=convnet_collate_fn) #collate_fn=collate_fn)
test_dataloader = DataLoader(list(zip(x_test, y_test)), batch_size=128, shuffle=True, collate_fn=convnet_collate_fn) # collate_fn=collate_fn)

#img_size = train_dataloader.dataset[0][0].shape

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
n_epochs = 20

model.train()
model.to('mps')
for epoch in range(n_epochs):
    print("epoch", epoch)
    epoch_train_loss = []
    for i, (images, labels) in tqdm.tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        images = images.to('mps')
        labels = labels.to('mps')
        print("images.shape", images.shape, images.dtype)
        #print("labels.shape", labels.shape, labels.dtype)
        out = model(images) #original_img_size = img_size)
        raise Exception
        loss = criterion(out, labels)
        #print("loss", loss.item())
        epoch_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    print("epoch_train_loss", np.mean(epoch_train_loss))


    with torch.no_grad():
        # testset
        epoch_test_loss = []
        model.eval()
        for i, (images, labels) in tqdm.tqdm(enumerate(test_dataloader)):
            images = images.to('mps')
            labels = labels.to('mps')
            out = model(images)
            loss = criterion(out, labels)
            epoch_test_loss.append(loss.item())
        print("epoch_test_loss", np.mean(epoch_test_loss))
    model.train()
 
#save model

torch.save(model.state_dict(), f"{model._name}_n_epoch_20.pt")
    

""" img_bytes = img_to_bytes(np.array(np_img))
print("img_bytes.shape", img_bytes.shape)
cfg = Megabyte_Config()
model = Megabyte(cfg)
#model = Patch_Embedder(cfg)
out = model(img_bytes)
print("out.shape", out.shape) 
 """