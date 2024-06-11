import numpy as np
import tensorflow as tf
import pickle
import os
import time
import argparse
import scipy
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

# 忽略 TensorFlow 的警告信息
tf.get_logger().setLevel('ERROR')
import numpy as np
from scipy.spatial.distance import mahalanobis


def filter_candidate_prototype_set(candidate_set, class_num, prototype_num_per_class):
    final_prototype_set = []

    for k in range(class_num):
        Ck = candidate_set[k]
        Nk = len(Ck)

        # Step 2: Compute Mahalanobis distance matrix
        Z = np.cov(Ck.T)
        Dk = np.zeros((Nk, Nk))
        for i in range(Nk):
            for j in range(Nk):
                Dk[i, j] = mahalanobis(Ck[i], Ck[j], np.linalg.inv(Z))

        # Step 3: Initialize array E
        E = np.max(Dk, axis=1)

        # Step 4-6: Update array E
        for i in range(Nk):
            for j in range(Nk):
                if j != i:
                    r = np.random.rand()  # This is just a placeholder, replace with actual r(x_j) > r(x_i)
                    if r > 0.5:  # Example condition
                        E[i] = min(E[i], Dk[i, j])

        # Step 7: Sort E in descending order
        Eind = np.argsort(-E)

        # Step 8-11: Add samples to the final prototype set
        Pk = []
        while len(Pk) < prototype_num_per_class:
            Pk.append(Ck[Eind[0]])
            Eind = Eind[1:]

        final_prototype_set.append(Pk)

    return final_prototype_set
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
def singnal_normalize_hp(data):
    signal_x = butter_highpass_filter(data, 30, 2500, 5)

    signal = signal_x * np.sqrt(1.0 / np.mean(np.abs(signal_x) ** 2))

    return signal


# 定义距离计算函数
def distance(features, centers):
    f_2 = tf.reduce_sum(tf.pow(features, 2), axis=1, keepdims=True)
    c_2 = tf.reduce_sum(tf.pow(centers, 2), axis=1, keepdims=True)
    dist = f_2 - 2 * tf.matmul(features, centers, transpose_b=True) + tf.transpose(c_2, perm=[1, 0])
    #torch.cdist,两个向量的距离

    return dist

def reshape_dataset(dataset, height, width):
    new_dataset = []
    for k in range(0, dataset.shape[0]):
        new_dataset.append(np.reshape(dataset[k], [height*width,1]))         # height,width,channel
    return np.array(new_dataset)


# 定义softmax损失函数
def softmax_loss(logits, labels):
    # labels=np.argmax(labels, axis=1)
    labels = tf.cast(labels, dtype=tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


# 自定义模型评价函数
def evaluate_model(model, dataset):
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    centers = []
    features = []

    for inputs, labels in dataset:
        features_batch, outputs, centers_batch = model(inputs, training=False)
        loss = dce_loss(features_batch, labels, centers_batch, args.temp) + args.weight_pl * pl_loss(features_batch,
                                                                                                     labels,
                                                                                                    centers_batch)

        total_loss += tf.reduce_sum(loss)
        correct_predictions += evaluation(features_batch, labels, centers_batch)[0]
        total_samples += inputs.shape[0]

        centers.append(centers_batch.numpy()[:,:2])
        features.append(features_batch.numpy()[:,:2])

    mean_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return mean_loss, accuracy, np.concatenate(centers), np.concatenate(features)


# 定义distance based cross entropy loss (DCE)
def dce_loss(features, labels, centers, T):
    dist = distance(features, centers)
    logits = -dist / T
    mean_loss = softmax_loss(logits, labels)
    return mean_loss


# 定义prototype loss (PL)
def pl_loss(features, labels, centers):
    batch_num = tf.cast(tf.shape(features)[0], tf.float32)
    # print(features.shape)
    batch_centers = tf.gather(centers, labels)
    # print(batch_centers.shape)
    dis = features - batch_centers
    return tf.divide(tf.nn.l2_loss(dis), batch_num)


# 定义训练操作
def training(loss, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op

# 定义evaluation操作
def evaluation(features, labels, centers):
    dist = distance(features, centers)
    # print(dist)
    prediction = tf.argmin(dist, axis=1, name='prediction')
    correct = tf.equal(tf.cast(prediction, tf.int64), labels, name='correct')
    #添加features,centers

    return tf.reduce_sum(tf.cast(correct, tf.float32), name='evaluation'),tf.cast(centers,tf.float32),tf.cast(features,tf.float32)

# 构造原型（中心点）
def construct_center(features, num_classes):
    len_features = features.shape[1]
    centers = tf.Variable(tf.zeros([num_classes, len_features]), dtype=tf.float32)
    return centers

# initialize the prototype with the mean vector (on the train dataset) of the corresponding class

# 定义模型
class PrototypeConvolutionalNetwork(tf.keras.Model):
    def __init__(self, input_shape, nclasses, temperature=1.0, weight_pl=0.001):
        super(PrototypeConvolutionalNetwork, self).__init__()
        self.nclasses = nclasses
        self.temperature = temperature
        self.weight_pl = weight_pl

        self.conv1 = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=1,
                                            kernel_initializer='glorot_uniform', input_shape=input_shape)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)

        self.conv2 = tf.keras.layers.Conv1D(80, 3, padding='same', activation='relu',
                                            kernel_initializer='glorot_uniform')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        self.features_layer = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal')
        self.dropout4 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(nclasses, kernel_initializer='he_normal')
        self.softmax = tf.keras.layers.Activation('softmax')
        #构造原型中心点
        self.centers = tf.Variable(tf.zeros([self.nclasses, 32]), dtype=tf.float32)              #n_components

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x=self.dropout3(x)
        features = self.features_layer(x)
        x=self.dropout4(x)
        x = self.dense2(x)
        outputs = self.softmax(x)

        return features, outputs, self.centers


if __name__=="__main__" :
    # 模型保存
    if not os.path.exists('./results'):
        os.makedirs('./results')

    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "best_model.ckpt")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    parser.add_argument('--stop', type=int, default=128, help='stopping number')
    parser.add_argument('--decay', type=float, default=0.3, help='the value to decay the learning rate')
    parser.add_argument('--temp', type=float, default=1.0, help='the temperature used for calculating the loss')
    parser.add_argument('--weight_pl', type=float, default=0.001, help='the weight for the prototype loss (PL)')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id for use')
    parser.add_argument('--num_classes', type=int, default=3, help='the number of classes')
    parser.add_argument('--num_epochs', type=int, default=3, help='the number of epochs to train')
    parser.add_argument('--dataset_path', type=str, default='G:/外场样本', help='path to the dataset')
    args = parser.parse_args()
    FLAGS, unparsed = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)                    #cpu

    # 加载数据
    os.chdir(args.dataset_path)
    file_chdir = os.getcwd()
    file_npy = []

    dfs = []

    for root, dirs, files in os.walk(file_chdir):
        for file in files:
            if os.path.splitext(file)[-1] == '.csv':
                file_directionary = os.path.join(root, file)
                df = pd.read_csv(file_directionary, header=None)
                df = df.dropna(axis=0, how='any')
                dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)

    np.random.seed(2023)
    n_examples = result_df.shape[0]

    n_train = int(n_examples * 0.8)
    n_test = int(n_examples * 0.1)

    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    remaining_idx = list(set(range(0, n_examples)) - set(train_idx))
    val_idx = np.random.choice(remaining_idx, size=n_test, replace=False)
    test_idx = list(set(remaining_idx) - set(val_idx))

    X_train = result_df.iloc[train_idx, :-1].values
    y_train = result_df.iloc[train_idx, -1].values
    X_val = result_df.iloc[val_idx, :-1].values
    y_val = result_df.iloc[val_idx, -1].values
    X_test = result_df.iloc[test_idx, :-1].values
    y_test = result_df.iloc[test_idx, -1].values

    X_train, X_test,X_val = map(np.array, (X_train, X_test,X_val))
    Y_train, Y_test,Y_val = map(np.array, (y_train, y_test,y_val))


    X_train = singnal_normalize_hp(X_train)
    X_val = singnal_normalize_hp(X_val)
    X_test = singnal_normalize_hp(X_test)

    X_train = reshape_dataset(X_train, 2500, 1)
    X_val = reshape_dataset(X_val, 2500, 1)
    X_test = reshape_dataset(X_test, 2500, 1)


    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(1000).batch(args.batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(args.batch_size)


    print(X_train.shape)
    input_shape = (2500, 1)

    model=PrototypeConvolutionalNetwork(input_shape=(2500,1),nclasses=args.num_classes)
    model.build((None, *input_shape))
    model.summary()
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++
    # model = PCNNet(input_shape=(2500,1),nclasses=3)
    # # # 这里调用模型一次以确保它的构建是基于实际的数据
    # # dummy_input = tf.random.normal([args.batch_size, *input_shape])
    # # model(dummy_input)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++

    lrate = args.lr
    optimizer = tf.keras.optimizers.SGD(learning_rate=lrate, momentum=0.9, decay=1e-4)
    num_epochs = args.num_epochs


    T_loss=[]
    T_acc=[]


    # 初始化变量用于记录验证集上的最佳性能
    best_val_loss = float('inf')
    patience = 0  # 记录连续验证集性能没有改善的次数
    max_patience = 5  # 最大容忍连续性能没有改善的次数
    early_stop = False  # 标志是否提前停止训练

    #定义loss

    for epoch in range(num_epochs):
        # Training
        running_loss = 0.0
        correct_predictions = 0
        train_num=0
        for inputs, labels in train_dataset:
            with tf.GradientTape() as tape:
                features, outputs,centers = model(inputs, training=True)
                loss1 = dce_loss(features,labels, centers,args.temp)
                loss2 = pl_loss(features,labels,centers)
                loss = loss1+args.weight_pl*loss2

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            running_loss += tf.reduce_sum(loss)
            correct_predictions += evaluation(features, labels, centers)[0]
            train_num+=1

        epoch_loss = running_loss / train_num
        epoch_acc = correct_predictions / train_num
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        T_loss.append(epoch_loss)
        T_acc.append(epoch_acc)

        # Validation
        val_loss = 0.0
        correct_predictions = 0
        val_num = 0
        for inputs, labels in val_dataset:
            features, outputs, centers = model(inputs, training=True)
            loss1 = dce_loss(features, labels, centers, args.temp)
            loss2 = pl_loss(features, labels, centers)
            loss = loss1 + args.weight_pl * loss2

            val_loss += tf.reduce_sum(loss)
            correct_predictions += evaluation(features, labels, centers)[0]
            val_num+=1

        val_loss = val_loss / val_num
        val_acc = correct_predictions / val_num
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # 检查验证集上的性能是否有改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # 保存最佳模型
            # model.save_weights(checkpoint_prefix)


        # 检查是否需要提前停止训练
        if epoch - best_epoch > max_patience:
            print("Early stopping!")
            break
    print("模型保存")
    model.save_weights(checkpoint_prefix)

    # 画图
    plt.figure(1)
    plt.plot(T_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('loss.png')

    plt.figure(2)
    plt.plot(T_acc)
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.savefig('./results/acc.png')

    loss, acc ,centers,features=evaluate_model(model,val_dataset)
    print('测试集准确率 {:.10f}%'.format(acc * 100))
    features = np.array(features)
    features = np.reshape(features, (-1, 2))
    print(features.shape)

    centers = np.array(centers)
    print('centers')
    centers = np.reshape(centers, (-1, 2))
    print(centers.shape)
    # 特征，圆型画图
    plt.figure(3)

    print(Y_test.shape)
    # colours=ListedColormap(['r','b','g',])
    label = ['1', '2', '3']
    # colors=range(13)
    cm = plt.colormaps.get_cmap('tab20_r')

    scatter = plt.scatter(features[:, 0], features[:, 1], c=Y_test, s=5, alpha=0.5, cmap=cm)
    plt.legend(handles=scatter.legend_elements()[0], labels=label, loc=3, title="classes")
    plt.scatter(centers[:, 0], centers[:, 1], s=50, c='r', marker='*', alpha=0.5, label='c')

    # plt.show()
    plt.savefig('./scatter.png')

    # 加载保存的模型权重
    # model.load_weights(checkpoint_prefix)
    # # 评估模型并绘制混淆矩阵
    #
    # y_pred,_,__ = model.predict(X_test)
    # print(y_pred)
    #
    #
    # y_pred_classes = np.argmax(y_test, axis=1)
    # print(y_pred_classes)

    # # 混淆矩阵
    # cm = confusion_matrix(y_test, y_pred_classes)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Class {i}' for i in range(args.num_classes)])
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix')
    # plt.savefig('./results/confusion_matrix.png')
    # plt.show()
    #
    #















