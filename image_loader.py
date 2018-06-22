import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ImageLoader:
    def __init__(self, path, folders):
        images, labels = [], []

        for i, folder_name in enumerate(folders):
            directory = path + folder_name + '/'
            files = os.listdir(directory)
            label = np.array([0] * 10)
            label[i] = 1
            for file in files:
                try:
                    im = imageio.imread(directory + file)
                except:
                    print("Skip a corrupted file: " + file)
                    continue
                pixels = np.array(im.astype(float))
                images.append(pixels / 255.0)
                labels.append(label)

        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
        
        class train:
            def __init__(self):
                self.images = []
                self.labels = []
                self.batch_counter = 0
                self.num_examples = 0

            def get_num_examples(self):
                    return len(self.images)

            def next_batch(self, num):
                if self.batch_counter + num >= len(self.labels):
                    batch_images = self.images[self.batch_counter:]
                    batch_labels = self.labels[self.batch_counter:]
                    left = num - len(batch_labels)
                    batch_images.extend(self.images[:left])
                    batch_labels.extend(self.labels[:left])
                    self.batch_counter = left
                else:
                    batch_images = self.images[self.batch_counter:self.batch_counter+num]
                    batch_labels = self.labels[self.batch_counter:self.batch_counter+num]                  
                    self.batch_counter += num
                    
                return (batch_images, batch_labels)
                    
        class test:
            def __init__(self):
                self.images = []
                self.labels = []
                
        self.train = train()
        self.test = test()

        self.train.images = train_images
        self.train.labels = train_labels
        self.test.images = test_images
        self.test.labels = test_labels