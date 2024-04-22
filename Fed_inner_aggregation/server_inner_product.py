import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
from Models import MultinomialLogisticRegression as MLR
from torchvision import datasets, transforms
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-ds', '--dataset', type=str, default='mnist', help='the name of dataset')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    f = open("log.txt", 'a')

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    elif args['model_name'] == 'MLR':
        net = MLR()
    else:
        exit('Error: undefined model')

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    if args['dataset'] == 'mnist':
        myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
        testDataLoader = myClients.test_data_loader
        # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        # dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # # sample users
        # if args['IID'] == 1:
        #     dict_users = mnist_iid(dataset_train, num_in_comm)
        # else:
        #     dict_users = mnist_noniid(dataset_train, num_in_comm)
    elif args['dataset'] == 'femnist':
        pass
    elif args['dataset'] == 'fashion_mnist':
        pass
    elif args['dataset'] == 'cifar10':
        pass


    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    # print('--------------------------------------------', file=f)
    # print('global_parameters:', global_parameters, file=f)
    # print('---------------------------------------', file=f)

    # 训练
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))
        print("communicate round {}".format(i+1), file=f)

        # 随机选择客户端
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]


        # print('clients_in_comm:', clients_in_comm, file=f)

        # 聚合
        glob_grad = None
        clients_grad = None
        clients_parameters = None

        for client in tqdm(clients_in_comm):
            #给客户端发送训练参数并接收返回的本地梯度和本地模型参数
            local_parameters, local_grad = myClients.clients_set[client].inner_localUpdate(args['epoch'],
                                                                                           args['batchsize'], net,
                                                                                           loss_func, opti,
                                                                                           global_parameters)
            if clients_grad is None:
                clients_grad = {}
                clients_parameters = {}

            # 将多个张量合并成一个张量
            local_grad_tensor = torch.cat([grad.flatten() for grad in local_grad])
            clients_grad[client] = local_grad_tensor
            clients_parameters[client] = local_parameters
        # 计算全局梯度（所有客户端的本地梯度累加求均值）
            if glob_grad is None:
                glob_grad = local_grad_tensor
            else:
                # 累加全局梯度
                glob_grad = glob_grad + local_grad_tensor
                # glob_grad = torch.add(glob_grad, local_grad_tensor)
        glob_grad = torch.div(glob_grad, args['num_of_clients'])
        # print('----------------------------------------------', file=f)
        # print('clients_grad:', clients_grad, file=f)
        # print('clients_parameters:', clients_parameters, file=f)
        # print('-------------------------------------------------', file=f)

        # print('-------------------------', file=f)
        # print('glob_grad:', glob_grad, file=f)
        # print('----------------------------', file=f)

        # 全局梯度
        # for j in range(len(glob_grad)):
        #     glob_grad[j] = glob_grad[j] / len(clients_in_comm)

        # print('--------------------------------------', file=f)
        # print('glob_grad:', glob_grad, file=f)
        # print('-------------------------------------', file=f)

        # 计算内积
        inner_product = None  # 存储每个客户端的内积
        sum_inner_product = None  # 内积和
        for client in tqdm(clients_in_comm):
            if sum_inner_product is None:
                inner_product = {}
                inner_product[client] = torch.dot(clients_grad[client], glob_grad)
                sum_inner_product = []
                sum_inner_product = inner_product[client]

            else:
                inner_product[client] = torch.dot(clients_grad[client], glob_grad)
                sum_inner_product = sum_inner_product + inner_product[client]

        # print('-------------------------------------', file=f)
        # print('inner_product:', inner_product, file=f)
        # print('sum_inner_product', sum_inner_product, file=f)
        # print('-------------------------------------', file=f)

        # 内积聚合
        # 将客户端内积除以内积和得到一个加权系数，乘以模型参数并累加
        result = None

        for client in tqdm(clients_in_comm):
            if result is None:
                result = {}
                for key, val in clients_parameters[client].items():
                    result[key] = (inner_product[client] / sum_inner_product) * val
            else:
                for key, val in clients_parameters[client].items():
                    result[key] = result[key] + (inner_product[client] / sum_inner_product) * val


        # print('----------------------', file=f)
        # print('result:', result, file=f)
        # print('---------------------------------', file=f)


        for key in global_parameters:
            #global_parameters[key] = global_parameters[key] + result[key]
            global_parameters[key] = result[key]


        with torch.no_grad():
            net.load_state_dict(global_parameters, strict=True)
            sum_accu = 0
            num = 0
            for data, label in testDataLoader:
                data, label = data.to(dev), label.to(dev)
                preds = net(data)
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
            print('accuracy: {}'.format(sum_accu / num))
            print('accuracy: {}'.format(sum_accu / num), file=f)

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))

