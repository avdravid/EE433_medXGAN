#!/usr/bin/env python
# coding: utf-8

# #### GPU / CPU Device

# In[19]:


import torch


device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    
print(device)


# #### Data Preprocessing

# In[20]:


import torchvision
import os

class SquashTransform:

    def __call__(self, inputs):
        return 2 * inputs - 1


BATCH_SIZE = 32
data_train = torchvision.datasets.ImageFolder(
    #'/home/amildravid/Feature_Vis/Tumor_testing/64x64/2tumor/Training',
    '/home/amildravid/Feature_Vis/Tumor_testing/tumors_upsampled',
    transform=torchvision.transforms.Compose([
         torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        SquashTransform()
    ])
)


data_val = torchvision.datasets.ImageFolder(
   #'/home/amildravid/Feature_Vis/Tumor_testing/64x64/2tumor/Testing',
    '/home/amildravid/Feature_Vis/Tumor_testing/archive/Testing',
    transform=torchvision.transforms.Compose([
         torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        SquashTransform()
    ])
)

# #fix labels
# if os.path.exists("/home/amildravid/Feature_Vis/Tumor_testing/tumors_upsampled/no_tumor/.ipynb_checkpoints"):
#     import shutil
#     shutil.rmtree("/home/amildravid/Feature_Vis/Tumor_testing/tumors_upsampled/no_tumor/.ipynb_checkpoints")


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


# In[21]:


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


num_steps =  ceil(len(data_val) / BATCH_SIZE)

num_steps


# #### Global Dimensions

# In[22]:


# Number of channels in the training images.
# For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz1 = 1000
nz2 = 100

# Size of labels dimension
nl = 2


# #### Generator Design
# 
# Reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# In[23]:


class Generator(torch.nn.Module):

    def __init__(self):

        super(Generator, self).__init__()
        self.fc1 = torch.nn.Linear(nz1+nz2, 768)
        
        self.main = torch.nn.Sequential(
            
            torch.nn.ConvTranspose2d(
                in_channels= 768,
                out_channels= 384,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features= 384
            ),
            torch.nn.ReLU(
                inplace=True
            ),

            
            torch.nn.ConvTranspose2d(
                in_channels= 384,
                out_channels= 256,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=256
            ),
            torch.nn.ReLU(
                inplace=True
            ),

            
            torch.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=192,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features= 192
            ),
            torch.nn.ReLU(
                inplace=True
            ),

            
            torch.nn.ConvTranspose2d(
                in_channels= 192,
                out_channels= 3,
                kernel_size=8,
                stride=2,
                padding=0,
                bias=False
            ),
            torch.nn.Tanh()
          
        )

    def forward(self, inputs, condition_latent_vec):
        # Concatenate Noise and Condition
        
        cat_inputs = torch.cat(
            (inputs, condition_latent_vec),
            dim=1
        )
        cat_inputs = self.fc1(cat_inputs)
        
        # Reshape the latent vector into a feature map.
        cat_inputs = cat_inputs.unsqueeze(2).unsqueeze(3)
        
        return self.main(cat_inputs)


# #### Discriminator Design

# In[24]:


class Discriminator(torch.nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = torch.nn.Sequential(
            # input is (nc) x 128 x 128
            torch.nn.Conv2d(
                in_channels=nc,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),
            torch.nn.Dropout(0.5, inplace=False),

           
            
            
            
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=32
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),
            torch.nn.Dropout(0.5, inplace=False),

            
            
            
            
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=64
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),
            torch.nn.Dropout(0.5, inplace=False),
            
            
            

            
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=128
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ), 
            torch.nn.Dropout(0.5, inplace=False),

            
            
            
            
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=256
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ), 
            torch.nn.Dropout(0.5, inplace=False),
            
            
            
            
            
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=512
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ), 
            torch.nn.Dropout(0.5, inplace=False),
            
            
            
            
            
            
        )
        
        
        # Real / Fake Classifier
        self.police = torch.nn.Sequential(
            torch.nn.Linear(5*5*512, 1), 
            torch.nn.Sigmoid()
            
        )

    def forward(self, input):
        
        features = self.main(input)
        #print(features.shape)
        valid = self.police(features.view(features.shape[0], -1)).view(-1, 1)
        return valid


# # Classifier

# In[25]:


class Classifier(torch.nn.Module):

    def __init__(self):

        super(Classifier, self).__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            ), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(num_features= 16),
            torch.nn.MaxPool2d(2), 
            torch.nn.Dropout(0.2), 
            
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=6,
                stride=1,
                padding=0,
                bias=False
            ), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(num_features= 32),
            torch.nn.MaxPool2d(2), 
            torch.nn.Dropout(0.2), 
            
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=6,
                stride=1,
                padding=0,
                bias=False
            ), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(num_features= 32), 
            torch.nn.MaxPool2d(2), 
            torch.nn.Dropout(0.2), 
            
            )
        # Categorical Classifier
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(
                in_features= 32*4*4,
                out_features= nl,
                bias=True
            ),
#             torch.nn.ReLU(), 
#             torch.nn.Linear(
#                 in_features= 32,
#                 out_features= nl,
#                 bias=True
#             ),
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
        clf = self.clf(features.view(features.shape[0], -1))
        return clf, features


# #### Weight Initialization

# In[26]:


# weight initialization
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


# Initialize Models
netG = Generator().to(device)
netD = Discriminator().to(device)

netG.apply(weights_init)
netD.apply(weights_init)

Classifier = Classifier().to(device)
Classifier.eval()

pretrain_weight_path = '/home/amildravid/Feature_Vis/Tumor_testing/64x64/64x64transforms_tumor_VGG_pretrain.pytorch'
Classifier.load_state_dict(torch.load(pretrain_weight_path))

for parameter in Classifier.parameters():
    parameter.requires_grad = False


# #### Optimizer

# In[27]:


# Setup Adam optimizers for both G and D
optimizerD = torch.optim.Adam(
    netD.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizerG = torch.optim.Adam(
    netG.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)


# In[28]:


num_examples = 10

fixed_noise = torch.randn(
    num_examples, nz1
).to(device)

fixed_noise_train = torch.randn(
    5, nz1
).to(device)


real_labels = torch.ones(BATCH_SIZE, 1).to(device)
fake_labels = torch.zeros(BATCH_SIZE, 1).to(device)

c1 = torch.nn.BCELoss()
# c2 = torch.nn.CrossEntropyLoss()


# #### Categorical Cross-Entropy

# In[29]:


def c2(input, target):

    _, labels = target.max(dim=1)

    return torch.nn.CrossEntropyLoss()(input, labels)


# #### One Hot Encoding

# In[30]:


def encodeOneHot(labels):
    ret = torch.FloatTensor(labels.shape[0], nl)
    ret.zero_()
    ret.scatter_(dim=1, index=labels.view(-1, 1), value=1)
    return ret


fixed_conditions = encodeOneHot(
    torch.randint(
        0,
        nl,
        (4, 1)
    )
).to(device)


# In[31]:


def condition_to_latent_vec(conditions):
    latent_vecs = torch.zeros((conditions.shape[0], nz2))
    
    for i in range (conditions.shape[0]):
        if conditions[i] == 0:
            latent_vecs[i,:]= torch.zeros((1, nz2))
        else: 
            latent_vecs[i,:] = torch.randn((1, nz2))
            
    latent_vecs = latent_vecs.to(device)
    
    return latent_vecs
        


# In[32]:


conditions_ex = torch.randint(0,nl,(num_examples, 1))
fixed_conditions = condition_to_latent_vec(conditions_ex).to(device)
fixed_conditions_train_neg = torch.zeros((5,nz2)).to(device)
fixed_conditions_train_pos = torch.randn((5,nz2)).to(device)


# #### Train Discriminator

# In[33]:


def trainD(images, labels):

    real_images = images.to(device)
    real_conditions = encodeOneHot(labels).to(device)

    
    
    fake_conditions_unencoded = torch.randint(
            0,
            nl,
            (BATCH_SIZE, 1)
        )
    
    fake_conditions = encodeOneHot(
        fake_conditions_unencoded
    ).to(device)
   
    
    fake_conditions_latent_vec =  condition_to_latent_vec(fake_conditions_unencoded)
    

    fake_images = netG(
        torch.randn(
            BATCH_SIZE, nz1
        ).to(device),
        fake_conditions_latent_vec     
    )
    
    

    optimizerD.zero_grad()

    real_valid = netD(real_images)
    fake_valid = netD(fake_images)
    
    l_s = c1(
        real_valid, real_labels
    ) + c1(
        fake_valid, fake_labels
    )


    d_loss = l_s

    d_loss.backward()

    optimizerD.step()

    return d_loss


# #### Train Generator

# In[34]:


def trainG(labels):
    fake_conditions_latent_vec =  condition_to_latent_vec(labels)
    conditions = encodeOneHot(labels).to(device)

    z = torch.randn(
        BATCH_SIZE, nz1
    ).to(device)

    netG.zero_grad()

    sample = netG(z, fake_conditions_latent_vec)

    
    valid_outputs = netD(sample)
    clf_outputs,_ = Classifier(sample)

    ls = c1(valid_outputs, real_labels)
    lc = c2(clf_outputs, conditions)

    loss = 2*lc + ls
    

    loss.backward()

    optimizerG.step()

    return loss


# #### Restore Checkpoint

# In[35]:


# netG.load_state_dict(torch.load('/home/amildravid/Feature_Vis/128_tests/gen.pytorch'))
# netD.load_state_dict(torch.load('/home/amildravid/Feature_Vis/128_tests/discr.pytorch'))
# optimizerG.load_state_dict(torch.load('/home/amildravid/Feature_Vis/128_tests/optGen.pytorch'))
# optimizerD.load_state_dict(torch.load('/home/amildravid/Feature_Vis/128_tests/optDisc.pytorch'))


# #### Train Network

# In[44]:


import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

import warnings
warnings.filterwarnings("ignore")


for epoch in range(5000):

    d_loss = 0
    g_loss = 0
    
    for i, (images, labels) in enumerate(train_loader):

        if i == num_steps:
            break

        for k in range(1):

             d_loss += trainD(images, labels)

        g_loss = trainG(labels)

    
    if epoch % 1 == 0:
        print(
            "E:{}, G Loss:{}, D Loss:{}".format(
                epoch+1,
                g_loss / num_steps,
                d_loss / num_steps / 1
            )
        )

        

        generated_neg = netG(fixed_noise_train,fixed_conditions_train_neg).to(device)
        classifier_neg_results,_ = Classifier(generated_neg)
        classifier_neg_results = classifier_neg_results.detach().cpu().numpy()
        generated_neg = generated_neg.detach().cpu().view(-1,3,64,64)
        
        
        
        generated_pos = netG(fixed_noise_train,fixed_conditions_train_pos).to(device)
        classifier_pos_results,_ = Classifier(generated_pos)
        classifier_pos_results = classifier_pos_results.detach().cpu().numpy()
        generated_pos = generated_pos.detach().cpu().view(-1,3, 64, 64)

        fig=plt.figure(figsize=(15, 2))
        plt.title('Negative')
        plt.axis('off')
        for i in range(1,6):
            minifig= fig.add_subplot(1, 5, i)
            minifig.axis('off')
            #_, label = torch.max(fixed_conditions[i-1], dim = 0)
            #minifig.title.set_text('Label: {}'.format(label))
            image = np.transpose(generated_neg[i-1,:,:,:],(1,2,0))
            image = (image + 1)/2
            minifig.text(0,75, classifier_neg_results[i-1, :], size = 'small')
            minifig.imshow(image)
            
            
        fig=plt.figure(figsize=(15, 2))
        plt.title('Positive')
        plt.axis('off')
        for i in range(1,6):                 
            minifig= fig.add_subplot(1, 5, i)
            minifig.axis('off')
            #_, label = torch.max(fixed_conditions[i-1], dim = 0)
            #minifig.title.set_text('Label: {}'.format(label))
            image = np.transpose(generated_pos[i-1,:,:,:],(1,2,0))
            image = (image + 1)/2
            minifig.text(0,75, classifier_pos_results[i-1, :], size = 'small')
            minifig.imshow(image)
        
        plt.show()
        
        
        # Save checkpoint
        torch.save(netG.state_dict(), './gen64.pytorch')
        torch.save(netD.state_dict(), './discr64.pytorch')
        torch.save(optimizerG.state_dict(), './optGen.pytorch')
        torch.save(optimizerD.state_dict(), './optDisc.pytorch')
        


# In[36]:


netG.load_state_dict(torch.load('./gen64.pytorch'))
netD.load_state_dict(torch.load('./discr64.pytorch'))


# #### Visualize Result

# In[19]:


import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

fig=plt.figure(figsize=(16, 12))
plt.title('Generated Images')
plt.axis('off')

num_examples = 10
fixed_noise = torch.randn(num_examples, nz1).to(device)
fixed_conditions_unencoded = torch.randint(0,nl,(num_examples, 1))
conditions_latent =  condition_to_latent_vec(fixed_conditions_unencoded)
generated = netG(fixed_noise, conditions_latent)

#pass through discriminator to get labels
discr_results,_ = Classifier(generated)
print(discr_results)
_,discr_preds = torch.max(discr_results, axis = 1)
print(discr_preds.detach().cpu().numpy())

imgs = generated.detach().cpu().view(-1, 3, 64, 64)


for i in range(1,num_examples+1):
    minifig= fig.add_subplot(2, 5, i)
    minifig.axis('off')
    label = fixed_conditions_unencoded[i-1].item()
    minifig.title.set_text('Label: {}'.format(label))
    image = np.transpose(imgs[i-1,::,:],(1,2,0))
    image = (image + 1)/2
    plt.imshow(image)


plt.show()


# #### Latent Space Fixed Noise + Different Label

# In[20]:


from scipy.stats import truncnorm
def truncate_noise(size, truncation):
    '''
    Function for creating truncated noise vectors: Given the dimensions (n_samples, z_dim)
    and truncation value, creates a tensor of that shape filled with random
    numbers from the truncated normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        truncation: the truncation value, a non-negative scalar
    '''
    
    truncated_noise = truncnorm.rvs(-1*truncation, truncation, size=(size, nz1))
    
    return torch.Tensor(truncated_noise)


# In[21]:


# one_noise = torch.randn(
#     1, nz1
# ).to(device)
#one_noise = truncate_noise(1,0.5)

one_noise = torch.randn(
        1, nz1
    ).to(device)
repeated_noise = one_noise.repeat(4, 1).to(device)
diff_conds = torch.tensor([[0],[1], [1], [1]])
diff_z2 = condition_to_latent_vec(diff_conds)


# In[22]:


generated = netG(repeated_noise,diff_z2).detach().cpu().view(-1, 3, 64, 64)
Classifier.to('cpu')
discr_results,_ = Classifier(generated)
discr_results = discr_results.detach().cpu().numpy()
print(discr_results)

image_neg = generated[0,:,:,:]
image_pos1 = generated[1, :, :, :]
image_pos2 = generated[2, :, :, :]
image_pos3 = generated[3, :, :, :]


fig=plt.figure(figsize=(15, 4))
#plt.title('Generated Image from Same Latent Vector z1', y = 0.85)
plt.axis('off')




minifig= fig.add_subplot(1, 4, 1)
minifig.axis('off')
minifig.title.set_text('Label: 0\n' + str(discr_results[0]))

image_neg = np.transpose(image_neg,(1,2,0))
image_neg = (image_neg + 1)/2
plt.imshow(image_neg)


minifig= fig.add_subplot(1, 4, 2)
minifig.title.set_text('Label: 1\n' + str(discr_results[1]))
minifig.axis('off')
image_pos1 = np.transpose(image_pos1,(1,2,0))
image_pos1 = (image_pos1 + 1)/2
plt.imshow(image_pos1)


minifig= fig.add_subplot(1, 4, 3)
minifig.title.set_text('Label: 1\n'+str(discr_results[2]))
minifig.axis('off')
image_pos2 = np.transpose(image_pos2,(1,2,0))
image_pos2 = (image_pos2 + 1)/2
plt.imshow(image_pos2)

minifig= fig.add_subplot(1, 4, 4)
minifig.title.set_text('Label: 1\n'+str(discr_results[3]))
minifig.axis('off')
image_pos3 = np.transpose(image_pos3,(1,2,0))
image_pos3 = (image_pos3 + 1)/2
plt.imshow(image_pos3)


plt.show()


# In[24]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

#Confusion Matrix and Classification Report
Y_true = []
Y_pred = []
Y_score = []
Classifier.to(device)
with torch.no_grad():
    for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device, dtype = torch.float)
            labels = labels.to(device, dtype = torch.float)
            outputs,_ = Classifier(inputs)
            _,predicted = torch.max(outputs, 1)
            score = outputs[:,1]
           
            [Y_true.append(n) for n in labels.cpu().numpy()]
            [Y_pred.append(n) for n in predicted.cpu().numpy()]
            [Y_score.append(n) for n in score.cpu().numpy() ]
print(confusion_matrix(Y_true, Y_pred))
target_names = ['COVID-Neg', 'COVID-Pos']
print(classification_report(Y_true, Y_pred, target_names=target_names))

fpr, tpr, thresholds = roc_curve(Y_true, Y_score)
auc1 = auc(fpr, tpr)
print(auc1)


# # Evaluating Disentanglement

# In[23]:


correct = 0 
Classifier.eval()
for i in range(1000):

    netG.to(device)
    Classifier.to(device)
    one_noise = torch.randn(
        1, nz1
    ).to(device)


    repeated_noise = one_noise.repeat(4, 1)
    diff_conds = torch.tensor([[0],[1], [1], [1]])
    diff_z2 = condition_to_latent_vec(diff_conds).to(device)
    generated = netG(repeated_noise,diff_z2).view(-1, 3, 64, 64)
    discr_results,_ = Classifier(generated)
    _,preds = torch.max(discr_results, axis = 1) 
    correct += (preds == torch.Tensor([0,1,1,1]).to(device)).sum()
    
print(correct)


# # Feature Visualization

# In[37]:


im = torch.randn(1, nz1).to(device)


# In[38]:


data_test = torchvision.datasets.ImageFolder(
    '/home/amildravid/Feature_Vis/Tumor_testing/visualize_image_64',#'/home/amildravid/Feature_Vis/Tumor_testing/visualize_image'
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
         torchvision.transforms.Resize((64, 64)),
        SquashTransform()
    ])
)
test_loader = torch.utils.data.DataLoader(
    data_test,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# In[39]:


import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
image = next(iter(test_loader))

Classifier.to(device)
Classifier.eval()
plt.figure(figsize=(3, 3))
plt.axis("off")
plt.title("Image")

plt.imshow(
    np.transpose(
        torchvision.utils.make_grid(
            image[0].to(device),
            padding=10,
            normalize=True,
            pad_value=1,
            nrow=int(3 * sqrt(BATCH_SIZE) / 2)
        ).cpu(),
        (1,2,0)
    )
);


clf, _ = Classifier(image[0].to(device))
clf = clf[0,1]
print(clf)


# ### Perceptual Loss

# In[40]:


import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


# In[41]:


from torch.autograd import Variable

init_noisez1 = Variable(torch.zeros(
    1, nz1
).to(device), requires_grad = True)

init_noisez2 = Variable(torch.randn(
    1, nz2
).to(device), requires_grad = True)


# In[42]:


optim = torch.optim.Adam([init_noisez1, init_noisez2], lr=0.01, betas=(0.5, 0.999))  


# In[43]:


original_image = (image[0][0].to(device))
#mask = torch.ones([1,3,128,128]).to(device)
#mask[0,:,4:60,20:60] = 2


# In[ ]:


#netG.eval
netG.eval()
netD.eval()
netD.to(device)
netG.to(device)
Classifier.to(device)
loss_func = VGGPerceptualLoss().to(device)

for epoch in range(0,1000000):
    original_image = image[0].to(device)
    optim.zero_grad()
    sample = netG(init_noisez1,init_noisez2 ).to(device)
    sample = (sample.reshape([1,3,64,64]))
    result,_ = Classifier(sample) 
    prob = result[0,1]
    
    class_loss = c1(prob, clf)
    
    discr = netD(sample)
    #source_loss = c1(discr, torch.ones([1,1]).to(device))
    original_image =  (original_image.reshape([1,3,64,64]))
    
    #print(loss_func(sample, original_image))
    #loss = 1*loss_func(sample, original_image)+ 100 * torch.mean((original_image - sample)**2) + 10*class_loss
    #loss =  class_loss + 4*torch.mean((original_image - sample)**2)
    loss = torch.mean((original_image - sample)**2)
    #loss = c1(sample, original_image)
    
    print("E:", epoch+1, "loss:", loss.item())
    loss.backward()
    optim.step()
    
    if (epoch+1) % 100 == 0:
        reconstructed_image = netG(
        init_noisez1, init_noisez2
        ).detach().cpu().view(-1, 3,64, 64)
        
        reconstructed_image = reconstructed_image[0,]
        
        print(result)
        print(netD(sample))
        fig=plt.figure(figsize=(5, 5))
        plt.title('Reconstruction')
        plt.axis('off')




        minifig= fig.add_subplot(1, 2, 1)
        minifig.axis('off')
        minifig.title.set_text('Original' + "\n")
        original_image = original_image.cpu().view(3, 64, 64)
        original_image = (np.transpose(original_image,(1,2,0))+1)/2
        original_image = (original_image)
        plt.imshow(original_image)


        minifig= fig.add_subplot(1, 2, 2)
        minifig.title.set_text('Reconstructed')
        minifig.axis('off')
        reconstructed_image = np.transpose(reconstructed_image,(1,2,0))
        reconstructed_image = (reconstructed_image + 1)/2
        plt.imshow(reconstructed_image)

        plt.show()


# In[429]:


#init_noisez3 = Variable(torch.clone(init_noisez2).to(device), requires_grad = True)
init_noisez3 = Variable(torch.randn(
    1, nz1
).to(device), requires_grad = True)
init_noisez4 = Variable(torch.randn(
    1, nz2
).to(device), requires_grad = True)
#Variable(torch.randn(1, nz2).to(device), requires_grad = True)
optim = torch.optim.Adam([init_noisez3, init_noisez4], lr=0.1, betas=(0.5, 0.999))  

#netG.eval
netG.eval()
netD.eval()
netD.to(device)
netG.to(device)
Classifier.to(device)
loss_func = VGGPerceptualLoss().to(device)

for epoch in range(0,1000000):
    original_image = image[0].to(device)
    optim.zero_grad()
    sample = netG(init_noisez3,init_noisez4 ).to(device)
    sample = (sample.reshape([1,3,64,64]))
    result,_ = Classifier(sample) 
    prob = result[0,1]
    zero = torch.Tensor([0])[0].to(device)
    
    #source_loss = c1(discr, torch.ones([1,1]).to(device))
    
    
    #print(loss_func(sample, original_image))
    #loss = 1*loss_func(sample, original_image)+ 100 * torch.mean((original_image - sample)**2) + 10*class_loss
    #loss =  class_loss + 4*torch.mean((original_image - sample)**2)
    
    loss = torch.mean((original_image - sample)**2)
    
    
    
    #1*c1(prob,zero)#+1*torch.mean((sample - netG(init_noisez1, init_noisez2).detach().view(-1, 3,64, 64))**2)
    
    
    #loss = c1(sample, original_image)
    
    print("E:", epoch+1, "loss:", loss.item())
    loss.backward()
    optim.step()
    
    if (epoch+1) % 100 == 0:
        neg_image = netG(
        init_noisez3, init_noisez4
        ).detach().cpu().view(-1, 3,64, 64)
        
        neg_image = neg_image[0,]
        
        print(result)
        
        fig=plt.figure(figsize=(5, 5))
        plt.title('Reconstruction')
        plt.axis('off')




        minifig= fig.add_subplot(1, 2, 1)
        minifig.axis('off')
        minifig.title.set_text('Negative' + "\n")
        neg_image = neg_image.cpu().view(3, 64, 64)
        neg_image = (np.transpose(neg_image,(1,2,0))+1)/2
        plt.imshow(neg_image)

        
        
        pos_image = netG(
        init_noisez1, init_noisez2
        ).detach().cpu().view(-1, 3,64, 64)
        
        pos_image = pos_image[0,]

        minifig= fig.add_subplot(1, 2, 2)
        minifig.title.set_text('Positive')
        minifig.axis('off')
        pos_image = np.transpose(pos_image,(1,2,0))
        pos_image = (pos_image + 1)/2
        plt.imshow(pos_image)

        plt.show()


# In[430]:


def overlay_mask(im1, mask):
    
    out = np.zeros((im1.shape))
    
    trues = np.where(mask ==1)
    falses = np.where(mask !=1)
    
    
    out[ trues[0], trues[1], :] = 1
    out[ falses[0], falses[1], :] = im1[falses[0], falses[1], :]
    #out = out.astype('uint8')
    
   
    return out

 


# In[431]:


netG.cpu()
netD.cpu()
Classifier.cpu()



# Reconstructed Pos
reconstructed_image = netG(init_noisez1.cpu(), init_noisez2.cpu()).detach().cpu().view(-1, 3, 64, 64)
reconstructed_image = torchvision.transforms.Grayscale(num_output_channels=1)(reconstructed_image)
reconstructed_image = torch.cat((reconstructed_image,reconstructed_image,reconstructed_image),1)
discr_result_pos = netD(reconstructed_image)
print(discr_result_pos)
clfrecpos, _ = Classifier(reconstructed_image)
print(clf)
reconstructed_image = reconstructed_image[0,]
reconstructed_image = np.transpose(reconstructed_image,(1,2,0))

# Reconstructed Neg
reconstructed_image_neg = netG(init_noisez3.cpu(), init_noisez4.cpu()).detach().cpu().view(-1, 3, 64, 64)
discr_result_neg = netD(reconstructed_image_neg)
print(discr_result_neg)
clfrecneg, _ = Classifier(reconstructed_image_neg)
print(clf)
reconstructed_image_neg = reconstructed_image_neg[0,]
reconstructed_image_neg = np.transpose(reconstructed_image_neg,(1,2,0))



#original
original_image = image[0][0].view(-1,3,64,64).cpu()
discr_result_original = netD(original_image)
print(discr_result_original)
clforiginal, _ = Classifier(original_image)
print(clf)
original_image = (np.transpose(original_image[0,],(1,2,0))+1)/2





z2_features_reconstructed = (reconstructed_image - reconstructed_image_neg)#/reconstructed_image


fig=plt.figure(figsize=(15, 6))
plt.title('Reconstruction')
plt.axis('off')




minifig= fig.add_subplot(1, 4, 1)
minifig.axis('off')
minifig.title.set_text('Original Positive\n' + str(clforiginal.detach().cpu().numpy()[0]))
plt.imshow(original_image)




minifig= fig.add_subplot(1, 4, 2)
minifig.title.set_text('Reconstructed Positive\n' +str(clfrecpos.detach().cpu().numpy()[0] ))
minifig.axis('off')
reconstructed_image = (reconstructed_image + 1)/2
plt.imshow(reconstructed_image)







minifig= fig.add_subplot(1, 4, 3)
minifig.title.set_text('Reconstructed Negative\n'+str(clfrecneg.detach().cpu().numpy()[0]))
minifig.axis('off')
reconstructed_image_neg = (reconstructed_image_neg + 1)/2
plt.imshow(reconstructed_image_neg)



minifig= fig.add_subplot(1, 4, 4)
minifig.title.set_text('Normalized Difference')
minifig.axis('off')
z2_features_reconstructed = (z2_features_reconstructed + 1)/2
z2_features_reconstructed = z2_features_reconstructed/torch.max(z2_features_reconstructed)
z2_features_reconstructed = np.where(z2_features_reconstructed<0,0, z2_features_reconstructed)
plt.imshow(z2_features_reconstructed)









plt.show()


# In[388]:


from skimage import img_as_ubyte

diff = img_as_ubyte(z2_features_reconstructed.numpy())
print(diff.shape)


# In[389]:


import cv2
diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
print(diff.shape)
heatmap_img = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
print(heatmap_img.shape)


# In[414]:


superimposed_img = heatmap_img * 0.005 + original_image.numpy()
#plt.imshow(superimposed_img/np.max(superimposed_img))


# In[415]:


final_img = superimposed_img[:,:,::-1]
final_img = final_img/np.max(final_img)
plt.imshow(final_img)


# # Counterfactual

# In[393]:


perturbed = next(iter(test_loader))[0]
map_im = np.dot(heatmap_img[...,:3], [0.2989, 0.5870, 0.1140])/255
idxs = np.where(map_im < 0.7)

a = idxs[0]

b = idxs[1]
for i in range(len(a)):
    perturbed[0, :, a[i],b[i]] = torch.mean(perturbed)# =  0
perturbed_im = perturbed[0]
plt.imshow((np.transpose(perturbed_im, (1,2,0))+1)/2)

clf, _ = Classifier(perturbed)
print(clf)


# ## Latent Space Interpolation

# In[443]:


delta = 0.1


# In[444]:


fig=plt.figure(figsize=(24, 10))
#plt.title('Latent Space Interpolation')
fig.suptitle('Latent Space Interpolation', fontsize=20)
plt.axis('off')


for i in range(11):
    minifig= fig.add_subplot(2, 6, i+1)
    image = netG(init_noisez3.cpu()+delta*i*(init_noisez1.cpu()-init_noisez3.cpu()), init_noisez4.cpu()+delta*i*(init_noisez2.cpu()-init_noisez4.cpu())).detach().cpu().view(-1, 3, 64, 64)
    discr_result = netD(image).detach().numpy()
    class_result,_ = Classifier(image)
    class_result = class_result.numpy()
    minifig.axis('off')
    
    if (i==0):
        minifig.title.set_text("Negative:"+"\n"+"Class: "+str(class_result[0]))
    elif (i==10):
        minifig.title.set_text("Final Positive:"+"\n"+"Class: "+str(class_result[0]))
    else: 
        minifig.title.set_text("Class: "+str(class_result[0]))
    plt.imshow((np.transpose(image[0],(1,2,0))+1)/2)
  

 #difference image
minifig= fig.add_subplot(2, 6, 12)
minifig.title.set_text('Positive-Negative Difference')
minifig.axis('off')
pos = netG(init_noisez1.cpu(), init_noisez2.cpu()).detach().cpu().view(-1, 3, 64,64)
neg = netG(init_noisez3.cpu(), init_noisez4.cpu()).detach().cpu().view(-1, 3,64, 64)
diffimg = pos - neg
plt.imshow((np.transpose(diffimg[0],(1,2,0))+1)/2)


# # Integrated Gradients

# In[445]:


neg = netG(init_noisez3.cpu(), init_noisez4.cpu()).view(-1, 3, 64, 64).clone().detach().requires_grad_(True)
#neg = torch.tensor(neg, dtype=torch.float32, device='cpu', requires_grad=True)
output,_ = Classifier(neg)
output


# In[446]:


target_label_idx = torch.argmax(output, 1).item()
index = np.ones((output.size()[0], 1)) * target_label_idx
index = torch.tensor(index, dtype=torch.int64)
output = output.gather(1, index)


# In[447]:


output


# In[448]:


Classifier.zero_grad()
output.backward()


# In[449]:


plt.imshow((np.transpose(neg.grad[0],(1,2,0))+1)/2)


# In[450]:


fig=plt.figure(figsize=(24, 10))
#plt.title('Latent Space Interpolation')
fig.suptitle('Latent Space Interpolation Gradients', fontsize=20)
plt.axis('off')


for i in range(11):
    minifig= fig.add_subplot(2, 6, i+1)
    image = netG(init_noisez3.cpu()+delta*i*(init_noisez1.cpu()-init_noisez3.cpu()), init_noisez4.cpu()+delta*i*(init_noisez2.cpu()-init_noisez4.cpu())).detach().cpu().view(-1, 3, 64, 64).clone().detach().requires_grad_(True)
    output,_ = Classifier(image)
    
    target_label_idx = torch.argmax(output, 1).item()
    index = np.ones((output.size()[0], 1)) * target_label_idx
    index = torch.tensor(index, dtype=torch.int64)
    output = output.gather(1, index)
    
    
    
    Classifier.zero_grad()
    output.backward()
   
    
    minifig.axis('off')
    
    plt.imshow((np.transpose(image.grad[0],(1,2,0))+1)/2)


# In[451]:


fig=plt.figure(figsize=(24, 10))
#plt.title('Latent Space Interpolation')
fig.suptitle('Accumulating and Averaging Gradients', fontsize=20)
plt.axis('off')


m = 11


mask = torch.zeros((3,64,64))

for i in range(m):
    minifig= fig.add_subplot(2, 6, i+1)
    image = netG(init_noisez3.cpu()+delta*i*(init_noisez1.cpu()-init_noisez3.cpu()), init_noisez4.cpu()+delta*i*(init_noisez2.cpu()-init_noisez4.cpu())).detach().cpu().view(-1, 3, 64, 64).clone().detach().requires_grad_(True)
    output,_ = Classifier(image)
    
    target_label_idx = torch.argmax(output, 1).item()
    index = np.ones((output.size()[0], 1)) * target_label_idx
    index = torch.tensor(index, dtype=torch.int64)
    output = output.gather(1, index)
    
    
    
    Classifier.zero_grad()
    output.backward()
    mask = (mask+image.grad[0])#/(i+1)

    
    minifig.axis('off')
    
    plt.imshow((np.transpose(mask/(i+1),(1,2,0))+1)/2)


# In[452]:


final_mask = diffimg[0] * (mask/m)
print(torch.max(final_mask))
print(torch.min(final_mask))

plt.imshow((np.transpose(final_mask,(1,2,0))+1)/2)


# In[453]:


final_mask = torch.tensor(final_mask)
final_mask = final_mask/torch.max(final_mask).item()


plt.imshow((np.transpose(final_mask,(1,2,0))+1)/2)


# In[454]:


#final_mask = final_mask.numpy()
final_mask = np.transpose(final_mask,(1,2,0))
rgb_weights = [0.2989, 0.5870, 0.1140]
final_mask_gray = np.dot(final_mask[...,:3], rgb_weights)
print(final_mask_gray)


# In[455]:


plt.imshow((final_mask_gray), cmap='gray')#, vmin=0, vmax=1)


# In[456]:


len(np.where(abs(final_mask_gray) >= 0.1)[0])


# In[364]:


from skimage import img_as_ubyte

final_mask_gray = img_as_ubyte(final_mask_gray)
print(final_mask_gray.shape)


# In[365]:


import cv2
#mask = cv2.cvtColor(final_mask, cv2.COLOR_RGB2GRAY)
#print(mask.shape)
heatmap_img = cv2.applyColorMap(final_mask_gray, cv2.COLORMAP_JET)
print(heatmap_img.shape)


# In[366]:


superimposed_img = heatmap_img * 0.04+ original_image.numpy()


# In[367]:


final_img = superimposed_img[:,:,::-1]
final_img = final_img/np.max(final_img)
plt.imshow(final_img)


# # Integrating Gradients Pixel-wise

# In[433]:


base = torch.zeros((3,64,64))
plt.imshow(np.transpose(base, (1,2,0)))


# In[434]:


original_image.shape


# In[435]:


fig=plt.figure(figsize=(24, 10))
#plt.title('Latent Space Interpolation')
fig.suptitle('Pixelwise Interpolation', fontsize=20)
plt.axis('off')


m = 11


target = np.transpose(original_image.clone(), (2,0,1))
print(target.shape)
target = target.view(-1,3,64,64).detach().requires_grad_(True)
for i in range(m):
    minifig= fig.add_subplot(2, 6, i+1)
    
    
    print(i*delta)
    mask = (base + i*delta*target)
    print(torch.equal(mask,target))
    title,_ = Classifier(mask)
    title = title.detach().numpy()
    plt.title(title)
    
    minifig.axis('off')
    plt.imshow((np.transpose(mask[0].detach().numpy(),(1,2,0))+1)/2)


# In[436]:


fig=plt.figure(figsize=(24, 10))

fig.suptitle('Pixelwise Interpolation Gradients', fontsize=20)
plt.axis('off')

target = netG(init_noisez1.cpu(), init_noisez2.cpu()).view(-1, 3, 64, 64).clone().detach().requires_grad_(True)

for i in range(11):
    minifig= fig.add_subplot(2, 6, i+1)
    
    image = (base + i*delta*target).clone().detach().requires_grad_(True)
    
    output,_ = Classifier(image)

    target_label_idx = torch.argmax(output, 1).item()
    index = np.ones((output.size()[0], 1)) * target_label_idx
    index = torch.tensor(index, dtype=torch.int64)
    output = output.gather(1, index)
    
    
    
    Classifier.zero_grad()
    output.backward()
   
    
    minifig.axis('off')
    
    plt.imshow((np.transpose(image.grad[0],(1,2,0))+1)/2)


# In[437]:


fig=plt.figure(figsize=(24, 10))
#plt.title('Latent Space Interpolation')
fig.suptitle('Accumulating and Averaging Gradients', fontsize=20)
plt.axis('off')


m = 11


mask = torch.zeros((3,64,64))

for i in range(m):
    minifig= fig.add_subplot(2, 6, i+1)
    image = netG(init_noisez1.cpu(), delta*i*init_noisez2.cpu()).view(-1, 3, 64, 64).clone().detach().requires_grad_(True)
    output, _ = Classifier(image)
    
    target_label_idx = torch.argmax(output, 1).item()
    index = np.ones((output.size()[0], 1)) * target_label_idx
    index = torch.tensor(index, dtype=torch.int64)
    output = output.gather(1, index)
    
    
    
    Classifier.zero_grad()
    output.backward()
    mask = (mask+image.grad[0])#/(i+1)

    
    minifig.axis('off')
    
    plt.imshow((np.transpose(mask/(i+1),(1,2,0))+1)/2)


# In[438]:


final_mask =  (mask/m)
print(torch.max(final_mask))
print(torch.min(final_mask))

plt.imshow((np.transpose(final_mask,(1,2,0))+1)/2)


# In[439]:


final_mask = torch.tensor(final_mask)
final_mask = final_mask/torch.max(final_mask).item()


plt.imshow((np.transpose(final_mask,(1,2,0))+1)/2)


# In[440]:


#final_mask = final_mask.numpy()
final_mask = np.transpose(final_mask,(1,2,0))
rgb_weights = [0.2989, 0.5870, 0.1140]
final_mask_gray = np.dot(final_mask[...,:3], rgb_weights)
print(final_mask_gray)


# In[441]:


plt.imshow((final_mask_gray+1)/2, cmap='gray')#, vmin=0, vmax=1)


# In[442]:


len(np.where(abs(final_mask_gray) >= 0.1)[0])


# In[ ]:




