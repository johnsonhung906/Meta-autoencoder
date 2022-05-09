import  torch, os
import  numpy as np
from    datasets import MVTec
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from MiniImagenet import MiniImagenet
from meta import Meta


# def mean_confidence_interval(accs, confidence=0.95):
#     n = accs.shape[0]
#     m, se = np.mean(accs), scipy.stats.sem(accs)
#     h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
#     return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    maml = Meta(args, model='conv_autoencoder').to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    # train_data = MVTec(args.data_root, 
    #             mode='train', 
    #             k_shot=args.k_spt,
    #             k_query=args.k_qry,
    #             resize=args.imgsz,
    #             batchsz=10000)

    # test_data = MVTec(args.data_root, 
    #             mode='test', 
    #             k_shot=args.k_spt,
    #             k_query=args.k_qry,
    #             resize=args.imgsz,
    #             batchsz=100)

    train_data = MiniImagenet(root='../MAML-Pytorch/miniimagenet/', mode='train', n_way=1, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)

    test_data = MiniImagenet(root='../MAML-Pytorch/miniimagenet/', mode='test', n_way=1, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(train_data, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            # print(x_spt.size(), y_spt.size(), x_qry.size() ,y_qry.size())

            loss = maml(x_spt, x_qry)

            if step % 30 == 0:
                print('step:', step, '\ttraining loss:', loss)

            if step % 500 == 0:  # evaluation
                db_test = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
                loss_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    loss = maml.finetunning(x_spt, x_qry)
                    loss_all_test.append(loss)

                # [b, update_step+1]
                loss = np.array(loss_all_test).mean(axis=0).astype(np.float16)
                print('Test loss:', loss)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_root', type=str, help='root dir of dataset', default='../data/MVTec')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    # argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
