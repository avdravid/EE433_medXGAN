#!/usr/bin/env python
# coding: utf-8

# #### GPU / CPU Device

# In[1]:


import torch


device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    
print(device)


# #### Data Preprocessing

# In[2]:


import torchvision
import os

class SquashTransform:

    def __call__(self, inputs):
        return 2 * inputs - 1


BATCH_SIZE = 32
data_train = torchvision.datasets.ImageFolder(
    '/home/amildravid/Feature_Vis/128_tests/128_training',
    transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        SquashTransform()
    ])
)


data_val = torchvision.datasets.ImageFolder(
    '/home/amildravid/Feature_Vis/128_tests/128_val',
    transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        SquashTransform()
    ])
)

#fix validation labels
if os.path.exists("/home/amildravid/Feature_Vis/128_tests/128_val/.ipynb_checkpoints"):
    os.rmdir("/home/amildravid/Feature_Vis/128_tests/128_val/.ipynb_checkpoints")


print(len(data_train))
print(len(data_val))



train_loader = torch.utils.data.DataLoader(
    data_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0, 
    drop_last=True
)


val_loader = torch.utils.data.DataLoader(
    data_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


classes = data_train.classes
print(data_val.classes)
print(classes)


# In[3]:


from math import ceil

BATCH_SIZE = 32

train_loader = torch.utils.data.DataLoader(
    data_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0, 
    drop_last=True
)


val_loader = torch.utils.data.DataLoader(
    data_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


num_steps =  ceil(len(data_train) / BATCH_SIZE)

num_steps


# #### Global Dimensions

# In[4]:


# Size of labels dimension
nl = 2


# #### Discriminator Design

# In[5]:


from torchvision.models import vgg16
model = vgg16()
model


# In[6]:


class Discriminator(torch.nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()
        
        self.gradients = None
        self.vgg = vgg16(pretrained = True)
        
        
        
        self.features = self.vgg.features[:30]
        self.maxpool = self.vgg.features[30]
        self.avgpool = self.vgg.avgpool
        
        self.fc = torch.nn.Sequential (self.vgg.classifier[0], self.vgg.classifier[1], self.vgg.classifier[2], 
                                      self.vgg.classifier[3], self.vgg.classifier[4], self.vgg.classifier[5])
        # Categorical Classifier
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(
                in_features= 4096,
                out_features= 1024,
                bias=True
            ),
            torch.nn.ReLU(), 
            torch.nn.Linear(
                in_features= 1024,
                out_features= 256,
                bias=True
            ),
            torch.nn.ReLU(), 
            torch.nn.Linear(
                in_features= 256,
                out_features= 64,
                bias=True
            ),
            torch.nn.ReLU(), 
            torch.nn.Linear(
                in_features= 64,
                out_features= nl,
                bias=True
            ),
            torch.nn.Softmax(dim=1)
        )
        
       
    
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features(x)
    
    
    def forward(self, input):
        features = self.features(input)
        if input.requires_grad:
            h = features.register_hook(self.activations_hook)
        features_pooled = self.avgpool(self.maxpool(features))
        features_fc = self.fc(features_pooled.view(features_pooled.shape[0], -1))
        clf = self.clf(features_fc.view(features_fc.shape[0], -1))
        return clf, features


# #### Optimizer

# In[7]:


# Setup Adam optimizers for both G and D
netD = Discriminator()
netD = netD.to(device)
for param in netD.features.parameters():
    param.requires_grad = False

params = list(netD.fc.parameters()) + list(netD.clf.parameters())

optimizerD = torch.optim.Adam(
    params,
    lr=0.0002,
    betas=(0.5, 0.999)
)


# In[25]:


# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in netD.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in netD.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')


# #### Categorical Cross-Entropy

# In[8]:


def c2(input, target):

    _, labels = target.max(dim=1)

    return torch.nn.CrossEntropyLoss()(input, labels)


# #### One Hot Encoding

# In[9]:


def encodeOneHot(labels):
    ret = torch.FloatTensor(labels.shape[0], nl).to(device)
    ret.zero_()
    ret.scatter_(dim=1, index=labels.view(-1, 1), value=1)
    return ret


# #### Train Discriminator

# In[10]:


def trainD(images, labels):
    images = images.to(device)
    conditions = encodeOneHot(labels.to(device))

    

    optimizerD.zero_grad()

    real_clf,_ = netD(images)
    

    d_loss = c2(real_clf, conditions) 

    d_loss.backward()

    optimizerD.step()

    return d_loss


# #### Restore Checkpoint

# In[29]:


'''
import os


if os.path.exists('./Para-GAN_gen.pytorch'):

    netG.load_state_dict(torch.load('./Para-GAN_gen.pytorch'))

if os.path.exists('./Para-GAN_disc.pytorch'):

    netD.load_state_dict(torch.load('./Para-GAN_disc.pytorch'))

if os.path.exists('./Para-GAN_optGen.pytorch'):

    optimizerG.load_state_dict(torch.load('./Para-GAN_optGen.pytorch'))

if os.path.exists('./Para-GAN_optDisc.pytorch'):

    optimizerD.load_state_dict(torch.load('./Para-GAN_optDisc.pytorch'))


'''


# #### Train Network

# In[30]:


import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")



for epoch in range(100):

    d_loss = 0
    running_val_loss = 0
    for i, (images, labels) in enumerate(train_loader):

        if i == num_steps:
            break

        for k in range(1):

             d_loss += trainD(images, labels)

    
    if epoch % 1 == 0:
        print(
            "E:{}, D Loss:{}".format(
                epoch+1,
                d_loss / num_steps / 1
            )
        )

        Y_true = []
        Y_pred = []
        Y_score = []
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device, dtype = torch.float)
                labels = labels.to(device, dtype = torch.int64)
                outputs,_ = netD(inputs)
                labels1 = encodeOneHot(labels).to(device)
                loss = c2(outputs, labels1)
                running_val_loss += loss.item()
                _,predicted = torch.max(outputs, 1)
                score = outputs[:,1]
                
                [Y_true.append(n) for n in labels.cpu().numpy()]
                [Y_pred.append(n) for n in predicted.cpu().numpy()]
                [Y_score.append(n) for n in score.cpu().numpy() ]
        print("Val Loss: " + str(running_val_loss))
        print(confusion_matrix(Y_true, Y_pred))
        target_names = ['COVID-Neg', 'COVID-Pos']
        print(classification_report(Y_true, Y_pred, target_names=target_names))
        fpr, tpr, thresholds = roc_curve(Y_true, Y_score)
        auc1 = auc(fpr, tpr)
        print(auc1)
        
        # Save checkpoint
        #torch.save(netG.state_dict(), './Para-GAN_gen.pytorch')
        torch.save(netD.state_dict(), './VGG_pretrain.pytorch')
        #torch.save(optimizerG.state_dict(), './Para-GAN_optGen.pytorch')
        #torch.save(optimizerD.state_dict(), './Para-GAN_optDisc.pytorch')


# In[11]:


pretrain_weight_path = '/home/amildravid/Feature_Vis/128_tests/VGG_pretrain.pytorch'
netD.load_state_dict(torch.load(pretrain_weight_path))


# In[32]:


import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
#Confusion Matrix and Classification Report
Y_true = []
Y_pred = []
Y_score = []
with torch.no_grad():
    for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device, dtype = torch.float)
            labels = labels.to(device, dtype = torch.float)
            outputs,_ = netD(inputs)
            _,predicted = torch.max(outputs, 1)
            score = outputs[:,1]
           
            [Y_true.append(n) for n in labels.cpu().numpy()]
            [Y_pred.append(n) for n in predicted.cpu().numpy()]
            [Y_score.append(n) for n in score.cpu().numpy() ]
print(confusion_matrix(Y_true, Y_pred))
target_names = ['COVID-Neg', 'COVID-Pos']
print(classification_report(Y_true, Y_pred, target_names=target_names))


# In[33]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(Y_true, Y_score)
auc1 = auc(fpr, tpr)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc1))
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()


# In[55]:


import numpy as np
import matplotlib.pyplot as plt
data_test = torchvision.datasets.ImageFolder(
    '/home/amildravid/Feature_Vis/128_tests/Visualization_images/Pos/',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        SquashTransform()
    ])
)
test_loader = torch.utils.data.DataLoader(
    data_test,
    batch_size=1,
    shuffle=False,
    num_workers=0
)
netD.eval()
netD = netD.to('cpu')

img,_ = next(iter(test_loader))
img.requires_grad=True

pred,_ = netD(img)#.argmax(dim=1)
print(pred)




 
pred[:, pred.argmax(dim=1)[0]].backward()

# pull the gradients out of the model
gradients = netD.get_activations_gradient()

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
activations = netD.get_activations(img).detach()

#print(activations.shape)

# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] += pooled_gradients[i]
    
# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)
heatmap = heatmap.numpy()
# draw the heatmap
plt.matshow(heatmap.squeeze())


# In[56]:


import cv2
img = cv2.imread('/home/amildravid/Feature_Vis/128_tests/Visualization_images/Pos/Positive/best.png')
heatmapcolor = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmapcolor = np.uint8(255 * heatmapcolor)
heatmapcolor = cv2.applyColorMap(heatmapcolor, cv2.COLORMAP_JET)
superimposed_img = heatmapcolor * 0.4 + img

#cv2.imwrite('/home/amildravid/Feature_Vis/preliminary_tests/visualization_images/CAM_images/CAM_COVID.jpg', superimposed_img)


# In[57]:


final_img = superimposed_img[:,:,::-1]
final_img = final_img/np.max(final_img)
plt.imshow(final_img)


# In[60]:


actual =  next(iter(test_loader))
heatmap_copy = heatmap.copy()
perturbed = actual[0].clone()

#heatmap_copy = np.where(heatmap_copy>0.75, np.random.normal(), heatmap_copy )
heatmap_copy = cv2.resize(heatmap_copy, (perturbed.shape[2], perturbed.shape[3]))


idxs = np.where(heatmap_copy>0.75)
a = idxs[0]
b = idxs[1]
for i in range(len(a)):
    perturbed[0, :, a[i],b[i]] = torch.mean(perturbed)# =  0
perturbed_im = np.asarray(perturbed)[0]
plt.imshow((np.transpose(perturbed_im, (1,2,0))+1)/2)



perturbed = torchvision.transforms.Resize((64,64))(perturbed).view(-1,3,64,64)

pred,_ = netD(perturbed)
print(pred)


# # T-SNE + PCA

# In[172]:


data_loader = torch.utils.data.DataLoader(data_val, 1, num_workers= 0, shuffle = True)


netD.to(device)


preds = torch.Tensor().to(device)
real_features = torch.Tensor().to(device)
all_images = torch.Tensor().to(device)
all_labels = torch.Tensor().to(device)
for i, (images, labels) in enumerate(data_loader):
    images = images.to(device)
    labels = labels.to(device)
    
    onepred,one_real_features = netD(images)
    
    preds = torch.cat([preds, onepred])
    real_features = torch.cat([real_features, one_real_features])
    all_labels = torch.cat([all_labels, labels])





#preds,real_features = netD(all_images)
_, preds = torch.max(preds, axis = 1)
preds = preds.cpu().numpy().reshape(-1)
real_labels = np.array(all_labels.cpu()).reshape(-1)


# In[173]:


print(len(real_labels))


# In[174]:


from sklearn.manifold import TSNE
from matplotlib import cm


classes = ["Negative", "Positive"]

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(real_features.view(2993,-1).cpu().detach().numpy())



# In[175]:


cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))
num_categories = 2
for lab in range(2):
    
    indices1 = preds==lab 
    indices2 = preds== real_labels
    indices = indices1==indices2
    
    
    
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], label = classes[lab], alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
#plt.xlim([-50, 50])
#plt.ylim([-50,50])
plt.title("T-SNE for True Image Feature Embeddings")
plt.show()


# In[176]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(real_features.view(2993,-1).cpu().detach().numpy())
pca_proj = pca.transform(real_features.view(2993,-1).cpu().detach().numpy())

print(pca.explained_variance_ratio_)

cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))
num_categories = 2
for lab in range(2):
    #indices = test_predictions==lab
    indices1 = preds==lab 
    indices2 = preds== real_labels
    indices = indices1==indices2
    
    ax.scatter(pca_proj[indices,0],pca_proj[indices,1], label = classes[lab] ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.title("PCA: True Image Feature Embeddings")
plt.show()


# In[ ]:





# In[ ]:




