
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision
from models1 import Net
from model import Model
from load_dataset import Custom_MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path
from utils import EarlyStopping, visualize_error, Metric
from torchvision.models import resnet18, vgg11
import torchvision.datasets as datasets

import random

# first load dataset, apply pre-processing  and show a sample
def main(opt):
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    Path('.\\runs\\train').mkdir(exist_ok=True, parents=True)
    i = 0
    while Path.is_dir(Path(f'.\\runs\\train\\exp{i}')):
        i+=1
    Path(f'.\\runs\\train\\exp{i}\\weights\\').mkdir(exist_ok=True, parents=True)
    Path(f'.\\runs\\train\\exp{i}').mkdir(exist_ok=True, parents=True)
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(20),
        transforms.RandomAutocontrast(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.7654 , std=0.3983)])

    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.7771 , std=0.3936)])


    train = Custom_MNIST('.\\train\\', transform=transform_train)
    valid = Custom_MNIST('.\\valid\\', transform=transform_valid)

    
    print(f'training set length {len(train)}\nvalidation set length { len(valid)}')

    # demonstrate a sample image


    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    fig1 = plt.figure(figsize=(8., 8.))
    grid1 = ImageGrid(fig1, 111,  # similar to subplot(111)
                    nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    for ax, ax1 in zip(grid, grid1):
        # Iterating over the grid returns the Axes.
        sample = train[int(np.random.randint(0,len(train)-1))]
        sample1 = valid[int(np.random.randint(0,len(valid)-1))]
        
        img, lbl = sample[0].squeeze(0).cpu().detach().numpy(), sample[1]
        img1, lbl1 = sample1[0].squeeze(0).cpu().detach().numpy(), sample1[1]

        ax.imshow(img, cmap='gray')
        ax.set_title(lbl.item())

        ax1.imshow(img1, cmap='gray')
        ax1.set_title(lbl1.item())
    fig1.savefig(f'.\\runs\\train\\exp{i}\\valid-samples.png')
    fig.savefig(f'.\\runs\\train\\exp{i}\\train-samples.png')

    # see distribution of label (use class weighting) => data is imbalanced
    print('traininig labels distributions: ',train.data_dist())
    print('validation labels distributions: ',valid.data_dist())    
    per_cls = train.data_dist()
    class_weights = [len(train) / per_cls[i] for i in range(10)]
    # use class weighting to overcome imbalanced classes
    #class_weights = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1]
    sample_weights = [0] * len(train)
    for idx, (data, label) in enumerate(train):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True)


    
    trainset = DataLoader(train, batch_size= opt.batch_size, sampler=sampler)
    #trainset = DataLoader(train, batch_size= opt.batch_size, shuffle=True)

    validset = DataLoader(valid, batch_size= opt.batch_size, shuffle=False)



    criterion = nn.CrossEntropyLoss( )
    model = Model()
    #model = Net(32*32, 256, 128, 10)

    optimizer = optim.Adam(model.parameters(), lr= opt.lr0, weight_decay= opt.weight_decay, betas=(0.9, 0.98))
    #optimizer = optim.SGD(model.parameters(), lr = opt.lr0, momentum=0.9, weight_decay= opt.weight_decay)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr= opt.lrf,
                        epochs=opt.epochs, anneal_strategy='linear', steps_per_epoch= int(len(trainset)) )

    #lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer, 100, 0.5)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = model.to(device)
    print(model)
    min_valid_loss = np.inf
    print('Starting to train the model ...')
    early_stopping = EarlyStopping(tolerance=15, min_delta=1.5)
    train_losses, valid_losses = [], []
    
    f = open(f".\\runs\\train\\exp{i}\\hyp_scratch.txt", "a")
    f.write(f'lr0: {opt.lr0}\nlrf: {opt.lrf}\nweight_decay: {opt.weight_decay}\nbatch-size: {opt.batch_size}\nepochs: {opt.epochs} ')
    f.close()

    import time
    steps_loss  = 2

    for epoch in range(opt.epochs):# tqdm(range(opt.epochs)):
        start_time = time.time()
        running_loss = 0.0
        v_running_loss = 0.0
        train_loss = 0.0
        totalt=0
        metric1 = Metric(num_classes=10)
        metric2 = Metric(num_classes=10)
        model.train()
        for id, (xt, yt) in enumerate(trainset):

            xt, yt = xt.to(device), yt.to(device)

            target = model(xt)
            optimizer.zero_grad()

            loss = criterion(target,yt)
            loss.backward()
            optimizer.step()
            model.eval()

            # print statistics
            with torch.no_grad():
                running_loss += loss.item()
                if id % steps_loss == (steps_loss-1):
                    # print every steps_loss mini-batches
                    np_pred = np.argmax(target.cpu().numpy(),-1)
                    np_gt = yt.cpu().numpy()
                    acc = np.sum(np_pred == np_gt) / opt.batch_size
                    print("[Epoch {:2d} - Iter {:3d}] loss: {:.3f} acc: {:.3f}".format(epoch + 1, id + 1, running_loss / steps_loss, acc))
                    running_loss = 0.0

        make_pred_on_dataloader(model, validset)
    print('Finished Training')
    '''
            train_loss += loss.item() * xt.size(0)
            totalt += xt.size(0)

            _, predicted = target.max(1)
            metric1.update(predicted, yt)

            lr_scheduler.step()

        train_losses.append(train_loss)
        #lr_scheduler2.step()
        # Validation
        print(epoch+1 % 15)
        print(epoch+1 % 15 == 0)
        if epoch+1 % 15 == 0:
            print(metric2.confusion_matrix())
        model.eval()

        with torch.no_grad():
            valid_loss = 0.0
            totalv = 0
            for xv, yv in validset:
                xv, yv = xv.to(device), yv.to(device)
                target = model(xv)
                loss = criterion(target, yv)
                valid_loss += loss.item() * xv.size(0)
                totalv += xv.size(0)
                _, predicted = target.max(1)
                metric2.update(predicted, yv)
            valid_losses.append(valid_loss)


        if min_valid_loss > valid_loss:

            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), f'.\\runs\\train\\exp{i}\\weights\\last.pt')
        print(f'Epoch [{epoch+1}/{opt.epochs}]\n ')
        print(f'Training:\t\t[Precision]: { metric1.precision():.2f}\t\t [Recall]: { metric1.recall():.2f}\t\t [Acc]:{metric1.accuracy():.2f}\t\t [Loss]:{train_loss / len(train):.4f}\n')
        print(f'Validation:\t\t[Precision]: { metric2.precision():.2f}\t\t [Recall]: { metric2.recall():.2f}\t\t [Acc]:{metric2.accuracy():.2f}\t\t [Loss]:{valid_loss /len(valid):.4f}\n ----------------------\n')
        f1 = open(f".\\runs\\train\\exp{i}\\Train_Monitoring.txt", "a")
        f1.write(f'Epoch [{epoch+1}/{opt.epochs}]\n ')
        f1.write(f'Training:\t\t[Precision]: { metric1.precision():.2f}\t\t [Recall]: { metric1.recall():.2f}\t\t [Acc]:{metric1.accuracy():.2f}\t\t [Loss]:{train_loss / len(train):.4f}\n')
        f1.write(f'Validation:\t\t[Precision]: { metric2.precision():.2f}\t\t [Recall]: { metric2.recall():.2f}\t\t [Acc]:{metric2.accuracy():.2f}\t\t [Loss]:{valid_loss /len(valid):.4f}\n ----------------------\n')
        f1.close()
        
        #early_stopping(train_loss, valid_loss) 
        #if early_stopping.early_stop:
        #    print("We are at epoch:", epoch)
        #    break
    '''


    visualize_error(train_losses, valid_losses, f'.\\runs\\train\\exp{i}')

def make_pred_on_dataloader(net, val_dataloader):
    import time

    y_pred = []
    y_true = []

    correct = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            start_time =  time.time()
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Convert tensor from GPU to CPU to Numpy array to List
            y_pred += list(predicted.cpu().numpy())
            y_true += list(labels.cpu().numpy())

            elapsed_time = time.time() - start_time
            print('[Iter {:2d}/{:2d}] - Elapsed time = {:.3f}'.
                  format(idx+1, len(val_dataloader), elapsed_time))

    acc = correct / total
    print('Accuracy of the network on the dataloader images: {:.4f} '.format(acc))

    return y_true, y_pred

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='total batch size')
    parser.add_argument('--epochs', type=int, default=500, help= 'number of training epochs')
    parser.add_argument('--lr0', type=float, default= 1e-7 )
    parser.add_argument('--lrf', type=float, default= 2e-3, help= ' final onecycle learning rate ')
    parser.add_argument('--weight_decay', type= float, default= 0.001)
    opt = parser.parse_args()
    main(opt)
'''
#confusion matrix
    plt.figure(figsize=(8, 6))
 
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None,cmap="YlGnBu",fmt="d")
 
plt.title("Confusion Matrix"), plt.tight_layout()
 
plt.ylabel("True Class"), 
plt.xlabel("Predicted Class")
plt.show()
'''

'''

 I meet the same problem but I figure out the problem is because I used BatchNorm1d before Linear layer
 , maybe it scale the output too much so that the final Linear scale it up to nearly similar value.
'''



'''
i reduced the learning rate to 10^-5 and the outputs are now different
'''

'''
This problem may be due to the “batch normalization”.
 When you are evaluating your model, you should disable batch normalization. Use “model.eval()”
 when you want to evaluate the model (so batch normalization will be disabled) and
  use “model.train()” again when you want train the model.
'''

'''
Hi,
I faced similar issue trying to save using .pth. When I saved as .pt it worked fine.
Try this:
'''