from PIL import Image
import torch
import numpy as np
from utils import img_to_bytes, bytes_to_img, text_to_bytes, bytes_to_text, MnistDataloader, patch_images
from model import Megabyte, Megabyte_Config, Patch_Embedder, VisualTransformer, VisualTransformer_Config, ResNet, ResNet_Config, Bottleneck
from torch.utils.data import DataLoader, vit_collate_fn, convnet_collate_fn, megabyte_collate_fn
from functools import partial
import tqdm

#model = VisualTransformer(VisualTransformer_Config())


model = ResNet(Bottleneck, ResNet_Config())   #Megabyte(Megabyte_Config())
(x_train, y_train), (x_test, y_test) = MnistDataloader().load_data()


print("vit params", VisualTransformer(VisualTransformer_Config()).get_param_count())
print("megabyte params", Megabyte(Megabyte_Config()).get_param_count())
print("convnet params", model.get_param_count())


#img_size = train_dataloader.dataset[0][0].shape


models = ["megabyte", "convnet", "vit"]

for model_name in models:
    print("model_name", model_name,"\n\n\n")

    if model_name == "megabyte":
        model = Megabyte(Megabyte_Config())
        collate_fn = partial(megabyte_collate_fn, type="image")
    
    elif model_name == "vit":
        model = VisualTransformer(VisualTransformer_Config())
        collate_fn = partial(vit_collate_fn, vit=True)
    
    elif model_name == "convnet":
        model = ResNet(Bottleneck, ResNet_Config())
        collate_fn = partial(convnet_collate_fn, convnet=True)

    train_dataloader = DataLoader(list(zip(x_train, y_train)), batch_size=128, shuffle=True, collate_fn=collate_fn) #collate_fn=collate_fn)
    test_dataloader = DataLoader(list(zip(x_test, y_test)), batch_size=128, shuffle=True, collate_fn=collate_fn) # collate_fn=collate_fn)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    n_epochs = 8

    model.train()
    model.to('mps')
    for epoch in range(n_epochs):
        print("epoch", epoch)
        epoch_train_loss = []
        for i, (images, labels) in tqdm.tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            images = images.to('mps')
            labels = labels.to('mps')
            #print("\nimages.shape", images.shape, images.dtype)
            #print("labels.shape", labels.shape, labels.dtype)
            out = model(images) #original_img_size = img_size)
            #print("out.shape", out.shape)
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