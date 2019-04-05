import os
import numpy as np
from dataset import Dataset, onehot

def urlretrieve(url, dst):
  import os.path
  import urllib.request
  if os.path.isfile(dst):
    print("[datasets] <%s> already exists. Skipping download." % dst)
    return
  print("[datasets] Downloading <%s> from <%s>." % (dst, url))
  return urllib.request.urlretrieve(url, dst)

def get_omniglot_data(selected_omni_groups=set(range(1000)), flatten=True):
    os.makedirs("../data/", exist_ok=True)
    urlretrieve("https://bitbucket.org/tkusmierczyk/lcvi-lc/raw/master/src/datasets/omniglot.pickle",
                "../data/omniglot.pickle")

    import pickle
    pickle_in = open("../data/omniglot.pickle", "rb")
    omni_data = pickle.load(pickle_in)
    pickle_in.close()

    train_mask = np.array(omni_data['training_mask'])
    features = np.array(omni_data['images']).astype(np.float32)
    labels = np.array(omni_data['classes']).astype(np.int32)

    # Selecting subset of groups ########################################
    groups = np.array(omni_data['groups'])
    group_mask = np.array([(g in selected_omni_groups) for g in groups])

    train_mask = train_mask[group_mask]
    features = features[group_mask]
    labels = labels[group_mask]
    groups = groups[group_mask]

    # old class numbers to new class numbers
    label2update = dict((label, i) for i, label in enumerate(sorted(set(labels))))
    labels = np.array([label2update[l] for l in labels]).astype(np.int32)

    print("[get_omniglot_data] %i groups %i labels" % (len(set(groups)), len(set(labels))))
    #####################################################################

    if flatten:
        features = features.reshape(-1, 28 * 28)

    y_train = labels[train_mask]
    x_train = features[train_mask]

    y_test = labels[~train_mask]
    x_test = features[~train_mask]

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test=get_omniglot_data(flatten=False)

def select_classes(x_train, y_train, x_test, y_test, classes=1623):
    classes = set(range(classes))
    train_mask = np.array([(y in classes) for y in y_train])
    test_mask = np.array([(y in classes) for y in y_test])
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test=select_classes(x_train, y_train, x_test, y_test,
                                                classes=500)
print("omniglot train amount",len(x_train))
"""
from matplotlib import pyplot
for i in range(50):
    pyplot.imshow(x_train[i,:,:,[0,0,0]].transpose())
    pyplot.title("%s" % y_train[i])
    pyplot.show()
"""



y_train, y_test=map(onehot,(y_train,y_test))
omniglot = Dataset(x_train, y_train, x_test, y_test)

