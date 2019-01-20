import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import copy
import time
import csv
from scipy import spatial
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ggplot import *
import warnings

print ("PACKAGES LOADED")


def get_proper_images(raw):
    images = np.array(raw, dtype=float)
    images = images.reshape([-1, 3, 32, 32])
    return images / 255.


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile, 'r', encoding='UTF8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done!")
    print("Dictionary Lenth: %d" % len(model))
    # print ("Done." + str(len(model)) + " words loaded!")
    return model


def mt_batch_iterator(randidx, batch_size, data, classification_label, coarse_vec, fine_vec):
    return data[randidx], classification_label[randidx], coarse_vec[randidx], fine_vec[randidx]


def negative_sampling(randidx, fine_text_vec,
                      coarse_data, fine_data,
                      coarse_name, fine_name,
                      batch_size, negative_size):
    neg_randidx = np.random.randint(negative_size, size=batch_size)

    negative_fine_list = []
    for i in randidx:
        temp_negative_fine_label = copy.deepcopy(coarse_fine_dic[coarse_name[coarse_data[i]]])
        fine_label = copy.deepcopy(fine_name[fine_data[i]])
        temp_negative_fine_label.remove(fine_label)
        negative_fine_list.append(temp_negative_fine_label)

    batch_negative_fine_lables = []
    for i, neg_idx in enumerate(neg_randidx):
        neg_fine_label_name = copy.deepcopy(negative_fine_list[i][neg_idx])
        neg_fine_idx = fine_name.index(neg_fine_label_name)
        batch_negative_fine_lables.append(fine_text_vec[neg_fine_idx])

    return np.asarray(batch_negative_fine_lables)


CIFAR_PATH = 'D:/dataset/CIFAR'
cifar100_tr_path = os.path.join(CIFAR_PATH, 'cifar-100-python/train')
cifar100_te_path = os.path.join(CIFAR_PATH, 'cifar-100-python/test')
cifar100_meta_path = os.path.join(CIFAR_PATH, 'cifar-100-python/meta')
cifar100_tr = unpickle(cifar100_tr_path)
cifar100_te = unpickle(cifar100_te_path)
cifar100_meta = unpickle(cifar100_meta_path)

print('train: %s' % cifar100_tr.keys())
print('test: %s' % cifar100_te.keys())
print('meta: %s' % cifar100_meta.keys())

CIFAR100_TRAIN_IMAGE = get_proper_images(cifar100_tr[b'data'])  # 50,000 images
CIFAR100_TRAIN_FINE_LABEL = np.asarray(cifar100_tr[b'fine_labels'])  # numeric from 0 to 99
CIFAR100_TRAIN_COARSE_LABEL = cifar100_tr[b'coarse_labels']  # numeric from 0 to 19

CIFAR100_TEST_IMAGE = get_proper_images(cifar100_te[b'data'])  # 10,000 images
CIFAR100_TEST_LABEL = np.asarray(cifar100_te[b'fine_labels'])  # numeric from 0 to 99
CIFAR100_TEST_COARSE_LABEL = cifar100_te[b'coarse_labels']  # numeric from 0 to 19

CIFAR100_FINE_LABEL_NAMES = cifar100_meta[b'fine_label_names']  # string from 0 to 99
CIFAR100_COARSE_LABEL_NAMES = cifar100_meta[b'coarse_label_names']  # string from 0 to 19
# TOTAL_LABEL_NAMES = copy.deepcopy(FINE_LABEL_NAMES + COARSE_LABEL_NAMES) #string from 0 to 119

# label preprocessing
CIFAR100_FINE_LABEL_NAMES[1] = b'aquarium';
CIFAR100_FINE_LABEL_NAMES[41] = b'mower';
CIFAR100_FINE_LABEL_NAMES[47] = b'maple';
CIFAR100_FINE_LABEL_NAMES[52] = b'oak';
CIFAR100_FINE_LABEL_NAMES[56] = b'palm';
CIFAR100_FINE_LABEL_NAMES[58] = b'pickup';
CIFAR100_FINE_LABEL_NAMES[59] = b'pine';
CIFAR100_FINE_LABEL_NAMES[83] = b'pepper';
CIFAR100_FINE_LABEL_NAMES[96] = b'willow'
print(CIFAR100_FINE_LABEL_NAMES)

# Coarse fine dictionary
coarse_fine_dic = {}
coarse_fine_dic["aquatic"] = ["beaver", "dolphin", "otter", "seal", "whale"]
coarse_fine_dic["fish"] = ["aquarium", "flatfish", "ray", "shark", "trout"]
coarse_fine_dic["flowers"] = ["orchid", "poppy", "rose", "sunflower", "tulip"]
coarse_fine_dic["containers"] = ["bottle", "bowl", "can", "cup", "plate"]
coarse_fine_dic["vegetables"] = ["apple", "mushroom", "orange", "pear", "pepper"]
coarse_fine_dic["devices"] = ["clock", "keyboard", "lamp", "telephone", "television"]
coarse_fine_dic["furniture"] = ["bed", "chair", "couch", "table", "wardrobe"]
coarse_fine_dic["insects"] = ["bee", "beetle", "butterfly", "caterpillar", "cockroach"]
coarse_fine_dic["carnivores"] = ["bear", "leopard", "lion", "tiger", "wolf"]
coarse_fine_dic["outdoor"] = ["bridge", "castle", "house", "road", "skyscraper"]
coarse_fine_dic["scenes"] = ["cloud", "forest", "mountain", "plain", "sea"]
coarse_fine_dic["herbivores"] = ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"]
coarse_fine_dic["mammals"] = ["fox", "porcupine", "possum", "raccoon", "skunk"]
coarse_fine_dic["worms"] = ["crab", "lobster", "snail", "spider", "worm"]
coarse_fine_dic["people"] = ["baby", "boy", "girl", "man", "woman"]
coarse_fine_dic["reptiles"] = ["crocodile", "dinosaur", "lizard", "snake", "turtle"]
coarse_fine_dic["rodent"] = ["hamster", "mouse", "rabbit", "shrew", "squirrel"]
coarse_fine_dic["trees"] = ["maple", "oak", "palm", "pine", "willow"]
coarse_fine_dic["transportation"] = ["bicycle", "bus", "motorcycle", "pickup", "train"]
coarse_fine_dic["vehicles"] = ["mower", "rocket", "streetcar", "tank", "tractor"]

print("<<Coarse label list>>")
coarse_name = []
fine_name = []
for label in CIFAR100_COARSE_LABEL_NAMES:
    # print(label.decode('UTF8'))
    coarse_name.append(label)
CIFAR100_COARSE_LABEL_NAMES = copy.deepcopy(coarse_name)
print(CIFAR100_COARSE_LABEL_NAMES)

# modify coarse_name
coarse_name = copy.deepcopy(list(coarse_fine_dic.keys()))
print("<<Modified coarse label list>>")
print(coarse_name)

print("\n\n<<Fine label list>>")
for label in CIFAR100_FINE_LABEL_NAMES:
    # print(label.decode('UTF8'))
    fine_name.append(label.decode('UTF8'))
CIFAR100_FINE_LABEL_NAMES = copy.deepcopy(fine_name)
print(CIFAR100_FINE_LABEL_NAMES)

CIFAR100_TRAIN_COARSE_LABEL_NAMES_LIST = []
for i, coarse_label in enumerate(CIFAR100_TRAIN_COARSE_LABEL):
    CIFAR100_TRAIN_COARSE_LABEL_NAMES_LIST.append(CIFAR100_COARSE_LABEL_NAMES[coarse_label])
print(len(CIFAR100_TRAIN_COARSE_LABEL_NAMES_LIST))

CIFAR100_TRAIN_FINE_LABEL_NAMES_LIST = []
for i, fine_label in enumerate(CIFAR100_TRAIN_FINE_LABEL):
    CIFAR100_TRAIN_FINE_LABEL_NAMES_LIST.append(CIFAR100_FINE_LABEL_NAMES[fine_label])
print(len(CIFAR100_TRAIN_FINE_LABEL_NAMES_LIST))

CIFAR100_TEST_LABEL_NAMES_LIST = []
for i, fine_label in enumerate(CIFAR100_TEST_LABEL):
    CIFAR100_TEST_LABEL_NAMES_LIST.append(CIFAR100_FINE_LABEL_NAMES[fine_label])
print(len(CIFAR100_TEST_LABEL_NAMES_LIST))

CIFAR100_MEAN = []
for i in range(np.shape(CIFAR100_TRAIN_IMAGE)[1]):
    CIFAR100_MEAN.append(np.mean(CIFAR100_TRAIN_IMAGE[:, :, :, i]))
print("\nCIFAR TRAINING RGB MEAN VALUE: %s" % CIFAR100_MEAN)

cifar10_test_batch = os.path.join(CIFAR_PATH, 'cifar-10-batches-py/test_batch')
cifar10_meta = os.path.join(CIFAR_PATH, 'cifar-10-batches-py/batches.meta')
cifar10_tb = unpickle(cifar10_test_batch)
cifar10_meta = unpickle(cifar10_meta)

print("cifar10 test batch %s" % cifar10_tb.keys())
print("cifar10 meta %s" % cifar10_meta.keys())

CIFAR10_LABEL_NAMES = []
for i in cifar10_meta[b'label_names']:
    CIFAR10_LABEL_NAMES.append(i.decode('utf-8'))
print(CIFAR10_LABEL_NAMES)

CIFAR10_TEST_IMAGE = get_proper_images(cifar10_tb[b'data'])
length = 10
CIFAR10_TEST_LABEL = cifar10_tb[b'labels']

CIFAR10_TEST_LABEL_NAMES_LIST = []
for i in CIFAR10_TEST_LABEL:
    CIFAR10_TEST_LABEL_NAMES_LIST.append(cifar10_meta[b'label_names'][i].decode('utf-8'))
print(len(CIFAR10_TEST_LABEL_NAMES_LIST))

#PREPROCESSED_CIFAR10_TEST_IMAGE = mean_subtract(CIFAR10_TEST_IMAGE, CIFAR100_MEAN)
GLOVE_PATH = 'D:/dataset/Glove/glove.6B.200d.txt'
glove = loadGloveModel(GLOVE_PATH)


# cifar100 label vector
CIFAR100_FINE_LABEL_GLOVE_VEC = []
for i, fine_label in enumerate(CIFAR100_FINE_LABEL_NAMES):
    CIFAR100_FINE_LABEL_GLOVE_VEC.append(glove[fine_label])
print(len(CIFAR100_FINE_LABEL_GLOVE_VEC))

# cifar10 label vector
CIFAR10_FINE_LABEL_GLOVE_VEC = []
for i, label in enumerate(CIFAR10_LABEL_NAMES):
    CIFAR10_FINE_LABEL_GLOVE_VEC.append(glove[label])
print(len(CIFAR10_FINE_LABEL_GLOVE_VEC))



# Coarse Vector
coarse_text_vec = []
coarse_text_vec.append(glove["aquatic"]) #aquatic_mammals
coarse_text_vec.append(glove["fish"]) #fish
coarse_text_vec.append(glove["flowers"]) #flowers
coarse_text_vec.append(glove["containers"]) #food_containers
coarse_text_vec.append(glove["vegetables"]) # fruit_and_vegetables
coarse_text_vec.append(glove["devices"]) #household_electrical_devices
coarse_text_vec.append(glove["furniture"]) #household_furniture
coarse_text_vec.append(glove["insects"]) #insects
coarse_text_vec.append(glove["carnivores"]) #large_carnivores
coarse_text_vec.append(glove["outdoor"]) #large_man-made_outdoor_things
coarse_text_vec.append(glove["scenes"]) #large_natural_outdoor_scenes
coarse_text_vec.append(glove["herbivores"]) #large_omnivores_and_herbivores
coarse_text_vec.append(glove["mammals"]) #medium_mammals
coarse_text_vec.append(glove["worms"]) #non-insect_invertebrates
coarse_text_vec.append(glove["people"]) #people
coarse_text_vec.append(glove["reptiles"]) #reptiles
coarse_text_vec.append(glove["rodent"]) #small_mammals
coarse_text_vec.append(glove["trees"]) #trees
coarse_text_vec.append(glove["transportation"]) #vehicles_1
coarse_text_vec.append(glove["vehicles"]) #vehicles_2

coarse_text_vec = np.asarray(coarse_text_vec)

# Fine Vector
fine_text_vec = []
for fine_label_name in CIFAR100_FINE_LABEL_NAMES:
    fine_text_vec.append(glove[fine_label_name])
fine_text_vec = np.asarray(fine_text_vec)


# image preprocessing
#PREPROCESSED_CIFAR100_TRAIN_IMAGE = mean_subtract(CIFAR100_TRAIN_IMAGE, CIFAR100_MEAN)
#PREPROCESSED_CIFAR100_TEST_IMAGE = mean_subtract(CIFAR100_TEST_IMAGE, CIFAR100_MEAN)

CIFAR100_TRAIN_COARSE_GLOVE_VEC = []
for i in range(len(CIFAR100_TRAIN_COARSE_LABEL)):
    CIFAR100_TRAIN_COARSE_GLOVE_VEC.append(glove[coarse_name[CIFAR100_TRAIN_COARSE_LABEL[i]]])
print()
print(len(CIFAR100_TRAIN_COARSE_GLOVE_VEC))
CIFAR100_TRAIN_COARSE_GLOVE_VEC = np.asarray(CIFAR100_TRAIN_COARSE_GLOVE_VEC)


CIFAR100_TRAIN_FINE_GLOVE_VEC = []
for i in range(len(CIFAR100_TRAIN_FINE_LABEL)):
    CIFAR100_TRAIN_FINE_GLOVE_VEC.append(glove[CIFAR100_TRAIN_FINE_LABEL_NAMES_LIST[i]])
print()
print(len(CIFAR100_TRAIN_FINE_GLOVE_VEC))
CIFAR100_TRAIN_FINE_GLOVE_VEC = np.asarray(CIFAR100_TRAIN_FINE_GLOVE_VEC)

print("###########################")
print("Loading dataset is Finished")
print("###########################")

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'custom_1': [64, 128, 'M', 256, 512, 1024, 'A']
}


class Unseen_detector(nn.Module):
    def __init__(self, vgg_name):
        super(Unseen_detector, self).__init__()
        self._conv_layer = self._make_layers(cfg[vgg_name])
        self._embedding_layer = nn.Sequential(nn.Linear(1024, 512),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(512, 100))

    def forward(self, x):
        features = self._conv_layer(x)
        flatten_features = features.view(features.size(0), -1)
        image_embedded_vec = self._embedding_layer(flatten_features)
        return image_embedded_vec

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for output_channel in cfg:
            if output_channel == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif output_channel == 'A':
                layers += [nn.AvgPool2d(kernel_size=8, stride=1)]
            else:
                layers += [nn.Conv2d(in_channels, output_channel, kernel_size=3, padding=0),
                           nn.BatchNorm2d(output_channel),
                           nn.ReLU(inplace=True)]
                in_channels = output_channel
        return nn.Sequential(*layers)


class VGG_embedding_model(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_embedding_model, self).__init__()
        self._conv_layer = self._make_layers(cfg[vgg_name])
        self._embedding_layer = nn.Sequential(nn.Linear(1024, 512),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(512, 200))

    def forward(self, x):
        features = self._conv_layer(x)
        flatten_features = features.view(features.size(0), -1)
        image_embedded_vec = self._embedding_layer(flatten_features)
        return image_embedded_vec

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for output_channel in cfg:
            if output_channel == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif output_channel == 'A':
                layers += [nn.AvgPool2d(kernel_size=8, stride=1)]
            else:
                layers += [nn.Conv2d(in_channels, output_channel, kernel_size=3, padding=0),
                           nn.BatchNorm2d(output_channel),
                           nn.ReLU(inplace=True)]
                in_channels = output_channel
        return nn.Sequential(*layers)


class Fine_text_embedding_model(nn.Module):
    def __init__(self):
        super(Fine_text_embedding_model, self).__init__()
        self._embedding_layer = nn.Sequential(
            nn.Linear(200, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 200))

    def forward(self, x):
        text_embedded_vec = self._embedding_layer(x)
        return text_embedded_vec


unseen_detector = Unseen_detector('custom_1')
image_embedding_net = VGG_embedding_model('custom_1')
text_embbedding_net = Fine_text_embedding_model()

x = torch.randn(3, 3, 32, 32)
logits = unseen_detector(x)
image_embedded_vec = image_embedding_net(x)

device = torch.device('cuda:1')
x = torch.from_numpy(CIFAR100_TRAIN_FINE_GLOVE_VEC[0:3])
x = x.to(dtype=torch.float)
text_embedded_vec = text_embbedding_net(x)

print(logits.size())
print(logits.dtype)
# print(logits)
print(torch.max(logits, 1)[1])

print()
print(image_embedded_vec.size())
print(image_embedded_vec.dtype)
# print(embedded_vec)
print(torch.max(image_embedded_vec, 1)[1])

print()
print(text_embedded_vec.size())
print(text_embedded_vec.dtype)
# print(embedded_vec)
print(torch.max(text_embedded_vec, 1)[1])

# embedding_net_criterion = nn.KLDivLoss()
mse_criterion = nn.MSELoss()
triplet_criterion = nn.TripletMarginLoss(margin=2.0, p=2, reduction='elementwise_mean')

image_embedding_net_optimizer = optim.SGD(image_embedding_net.parameters(), lr=1e-2, momentum=0.8)
text_embedding_net_optimizer = optim.SGD(text_embbedding_net.parameters(), lr=1e-2, momentum=0.8)

device = torch.device('cuda:1')
image_embedding_net.to(device)
text_embbedding_net.to(device)
# image_embedding_net = nn.DataParallel(image_embedding_net)
# text_embbedding_net = nn.DataParallel(text_embbedding_net)

print(image_embedding_net)
print()
print(text_embbedding_net)

epochs = 201
display_step = 1
save_step = 10
batch_size=256
num_train_iteration = int(len(CIFAR100_TRAIN_IMAGE)/batch_size)
warnings.filterwarnings(action='once')
for epoch in range(epochs):
    start_time = time.time()
    if epoch==0:
        print('\n===> epoch %d' % epoch)



    running_image_coarse_mse_loss = 0.0
    running_text_coarse_mse_loss = 0.0
    running_image_text_triplet_loss = 0.0

    running_image_loss = 0.0
    running_text_loss = 0.0
    running_total_loss = 0.0

    for i in range(num_train_iteration):
        # get the inputs
        randidx = np.random.randint(CIFAR100_TRAIN_IMAGE.shape[0], size=batch_size)
        batch_inputs, batch_cls_label, batch_coarse_vec, batch_fine_vec = mt_batch_iterator(randidx = randidx,
                                                                                            batch_size = batch_size,
                                                                                            data = CIFAR100_TRAIN_IMAGE,
                                                                                            classification_label = CIFAR100_TRAIN_FINE_LABEL,
                                                                                            coarse_vec = CIFAR100_TRAIN_COARSE_GLOVE_VEC,
                                                                                            fine_vec = CIFAR100_TRAIN_FINE_GLOVE_VEC)


        batch_neg_fine_vec = negative_sampling(randidx = randidx, fine_text_vec = fine_text_vec,
                                               coarse_data = CIFAR100_TRAIN_COARSE_LABEL,
                                               fine_data = CIFAR100_TRAIN_FINE_LABEL,
                                               coarse_name = coarse_name,
                                               fine_name = fine_name,
                                               batch_size = batch_size,
                                               negative_size = 4)

        batch_inputs = torch.from_numpy(batch_inputs)
        batch_cls_labels = torch.from_numpy(batch_cls_label)
        batch_coarse_vec = torch.from_numpy(batch_coarse_vec)
        batch_fine_vec = torch.from_numpy(batch_fine_vec)
        batch_neg_fine_vec = torch.from_numpy(batch_neg_fine_vec)


        batch_inputs = batch_inputs.to(device=device, dtype=torch.float)
        batch_cls_labels = batch_cls_labels.to(device=device, dtype=torch.float)
        batch_coarse_vec = batch_coarse_vec.to(device=device, dtype=torch.float)
        batch_fine_vec = batch_fine_vec.to(device=device, dtype=torch.float)
        batch_neg_fine_vec = batch_neg_fine_vec.to(device=device, dtype=torch.float)

        # zero the parameter gradients
        image_embedding_net_optimizer.zero_grad()
        text_embedding_net_optimizer.zero_grad()

        # forward
        img_emb_outputs = image_embedding_net(batch_inputs)
        txt_pos_emb_outputs = text_embbedding_net(batch_fine_vec)
        txt_neg_emb_outputs = text_embbedding_net(batch_neg_fine_vec)

        #loss
        image_coarse_mse_loss = mse_criterion(img_emb_outputs, batch_coarse_vec)
        text_coarse_mse_loss = mse_criterion(txt_pos_emb_outputs, batch_coarse_vec)
        image_text_triplet_loss = triplet_criterion(img_emb_outputs, txt_pos_emb_outputs, txt_neg_emb_outputs)

        #print("<<mse>>")
        #print(image_coarse_mse_loss)
        #print("<<triplet>>")
        #print(image_text_triplet_loss)
        #print(np.shape(image_text_triplet_loss))

        alpha = 0.5
        beta = 0.5
        image_loss = torch.add((1-alpha)*image_coarse_mse_loss, alpha*image_text_triplet_loss)
        text_loss = torch.add((1-beta)*text_coarse_mse_loss, beta*image_text_triplet_loss)
        #total_loss = torch.add(image_loss, text_loss)

        #backward + optimize
        image_loss.backward(retain_graph=True)
        image_embedding_net_optimizer.step()

        #backward + optimize
        text_loss.backward(retain_graph=False)
        text_embedding_net_optimizer.step()

        total_loss = image_loss + text_loss

        # print statistics
        running_image_coarse_mse_loss += image_coarse_mse_loss.item()
        running_text_coarse_mse_loss += text_coarse_mse_loss.item()
        running_image_text_triplet_loss += image_text_triplet_loss.item()

        running_image_loss += image_loss.item()
        running_text_loss += text_loss.item()
        running_total_loss += total_loss.item()

    end_time = time.time()
    elapsed_time = end_time - start_time
    if epoch % display_step == 0:    # print every displaystep

    ## Let us look at how the network performs on the whole dataset.
    #    correct = 0.
    #    total = 0.
    #    num_test_iteration = int(len(CIFAR100_TEST_IMAGE)/batch_size)
    #    with torch.no_grad():
    #        start_idx = 0
    #        end_idx = batch_size
    #        for i in range(num_test_iteration):
    #            inputs = CIFAR100_TEST_IMAGE[start_idx:end_idx]
    #            labels = CIFAR100_TEST_LABEL[start_idx:end_idx]
    #            inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
    #            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
    #            outputs, _ = embedding_net(inputs.cuda())
    #            _, predicted = torch.max(outputs.data, 1)
    #            total += float(labels.size(0))
    #            correct += float((predicted == labels).sum().item())
    #            start_idx = copy.deepcopy(end_idx)
    #            end_idx = end_idx + batch_size
    #
    #    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        print('[epoch:%d] I-C_mse_loss: %.4f \tT-C_mse_loss: %.4f \tI-T_triplet_loss: %.4f \nimage_loss: %.4f \ttext_loss: %.4f \ttotal_loss: %.4f  \telapsed time: %.2fs/epoch' % (epoch+1,
                                                                                                             (running_image_coarse_mse_loss / num_train_iteration),
                                                                                                             (running_text_coarse_mse_loss / num_train_iteration),
                                                                                                             (running_image_text_triplet_loss / num_train_iteration),
                                                                                                             (running_image_loss / num_train_iteration),
                                                                                                             (running_text_loss / num_train_iteration),
                                                                                                             (running_total_loss / num_train_iteration),
                                                                                                             elapsed_time))

    if epoch % save_step == 0:    # print every displaystep
        # save model at each epoch
        image_model_path = './embedding_net/image_embedding_net_'+str(epoch)+'.pth'
        text_model_path = './embedding_net/text_embedding_net_'+str(epoch)+'.pth'
        torch.save(image_embedding_net, image_model_path)
        torch.save(text_embbedding_net, text_model_path)


    running_loss = 0.0

print("Training is finished")