import torch.nn as nn
import numpy as np
from model import Model
import torch
from pathlib import Path
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, vgg11
import torch.nn.functional as F

class inference():
    def __init__(self, modelPath):

        self.modelPath = modelPath
        self.outx = []
    def get_dir_id(self):

        i = 0
        while Path.is_dir(Path(f'.\\runs\\inference\\infer{i}')):
            i+=1
        Path(f'.\\runs\\inference\\infer{i}\\').mkdir(exist_ok=True, parents=True)
        return i

    def getScores(self, images:np.ndarray):

        Path(f'.\\runs\\inference\\').mkdir(exist_ok=True, parents=True)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = Model()
        model.load_state_dict(torch.load(self.modelPath,map_location=torch.device('cpu')))
        model = model.to(device)
        id = self.get_dir_id()
        transform_valid = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.7771 , std=0.3936)])

        model.eval()

        for idx, img1 in enumerate(images):

            img = transform_valid(img1).unsqueeze(0)
            img = img.to(device)
            print(img.shape)
            out = model(img)
            
            #pred = F.softmax(out.max(1, keepdim=True)[1].float()).item()
            pred = (torch.argmax(out)).item()
            #print(pred)
            print(out)
            self.outx.append(out)
            img = img.view(32,32).cpu().numpy()
            #cv2.imshow('r', img1)
            #cv2.waitKey(0)
            cv2.imwrite(f'.\\runs\\inference\\infer{id}\\{pred}_{idx}.jpg', img1)
        self.outx = torch.stack(self.outx, 0).squeeze(1)
        return self.outx


import glob
images = glob.glob('.\\valid\\*.jpg')
imgs  = [cv2.imread(img, 0) for img in images]
obj = inference(modelPath='.\\runs\\train\\exp33\\weights\\last.pt')
out = obj.getScores(imgs)
print(out.shape)








'''
def getScores1(self, images:np.ndarray):

    Path(f'.\\runs\\inference\\').mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model()
    #model = vgg11()
    #num_features = model.classifier[6].in_features 
    #features = list(model.classifier.children())[:-1] # Remove last layer
    #features.extend([nn.Linear(num_features, 10)])
    #model.classifier = nn.Sequential(*features)
    #model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    id = self.get_dir_id()
    with torch.no_grad():
        for idx, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (32, 32))
            img1 = torch.from_numpy(img).to(device)
            img1 = img1.float()

            img1 = img1.view(1,1,32,32)

            out = model(img1)
            pred = (torch.argmax(out)).item()

            
            self.outx.append(out)
            cv2.imwrite(f'.\\runs\\inference\\infer{id}\\{pred}_{idx}.jpg', img)
    self.outx = torch.stack(self.outx, 0).squeeze(1)
    return self.outx
'''
