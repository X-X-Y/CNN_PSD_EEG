import paddle as paddle
import numpy as np
import os
import paddle.nn.functional as F
from paddle.io import Dataset
from paddle.static import InputSpec
from visualdl import LogWriter

data_path = 'Your DataDir'
label_path = 'Your LeabelDir'
datas = np.load(data_path).reshape(6520, 1, 3, 129)
labels = np.load(label_path).reshape(6520, 1)
print(datas.shape, labels.shape)

# 定义data reader
class dataReader(Dataset):
    def __init__(self, datas, labels, mode='test'):
        super(dataReader, self).__init__()
        assert mode in ['train', 'test'], "mode should be 'train' or 'test', but got {}".format(mode)
        self.datas = []
        self.labels = []
        if mode == 'train' :
            datas = datas[:int(datas.shape[0]*0.8), :, :, :].astype('float32')
            labels = labels[:int(labels.shape[0]*0.8)].astype('int32')
            # print(mode, datas.shape)
        elif mode == 'test' :
            datas = datas[int(datas.shape[0]*0.8):, :, :, :].astype('float32')
            labels = labels[int(labels.shape[0]*0.8):].astype('int32')
            # print(mode, datas.shape)
        self.datas = datas
        self.labels = labels
    
    def __getitem__(self, index):
        data = paddle.to_tensor(self.datas[index].astype('float32'))
        label = paddle.to_tensor(self.labels[index].astype('int64'))
        return data, label
    
    def __len__(self):
        return len(self.datas)

BATCH_SIZE = 32
train_loader = dataReader(datas, labels, 'train')
test_loader = dataReader(datas, labels, 'test')
train_reader = paddle.io.DataLoader(train_loader, batch_size=BATCH_SIZE, shuffle=True)
test_reader = paddle.io.DataLoader(test_loader, batch_size=BATCH_SIZE)
print(train_loader)
print(test_loader)

# 网络结构
class DATA2Net(paddle.nn.Layer):
    def __init__(self, num_classes=2):
        super(DATA2Net, self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=32, kernel_size=(1, 3))
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=(1, 4), stride=(1, 4))

        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(2,3))
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=(1, 4), stride=(1, 4))

        self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1,3))

        self.flatten = paddle.nn.Flatten()

        self.linear1 = paddle.nn.Linear(in_features=640, out_features=128)
        self.dropout = paddle.nn.Dropout(0.3)
        self.linear2 = paddle.nn.Linear(in_features=128, out_features=num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

epoch_num = 300
learning_rate = 0.00001
def train(model):
	# 训练模式
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=learning_rate,
                                parameters=model.parameters())
    with LogWriter(logdir='./log/train') as train_writer, LogWriter(logdir='./log/test') as test_writer:
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                x_data = data[0]
                y_data = paddle.to_tensor(data[1])
                # print(y_data)

                x_predict = model(x_data)
                # print(x_predict)
                loss = F.cross_entropy(x_predict, y_data)
                acc = paddle.metric.accuracy(x_predict, y_data)
                loss.backward()
                opt.step()
                opt.clear_grad()
            print("train_epoch: {}, loss is: {}, acc is: {}".
                format(epoch, loss.numpy(), acc.numpy()))
            train_writer.add_scalar(tag='train/loss', step=epoch, value=loss.numpy())
            train_writer.add_scalar(tag='train/acc', step=epoch, value=acc.numpy())
			
			# 验证模式
            model.eval()
            accs = []
            losses = []
            for batch_id, data in enumerate(test_reader):
                x_data = data[0]
                y_data = paddle.to_tensor(data[1])

                x_predict = model(x_data)
                loss = F.cross_entropy(x_predict, y_data)
                acc = paddle.metric.accuracy(x_predict, y_data)
                accs.append(acc.numpy())
                losses.append(loss.numpy())

            avg_acc, avg_loss = np.mean(accs), np.mean(losses)
            print("[test] accuracy/loss: {}/{}".format(avg_acc, avg_loss))
            test_writer.add_scalar(tag='test/loss', step=epoch, value=avg_loss)
            test_writer.add_scalar(tag='test/acc', step=epoch, value=avg_acc)
            model.train()

model = DATA2Net()
train(model)






