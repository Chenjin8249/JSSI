import argparse
import random
import torch.utils.data
# from ACtest import *
from ACmodel import *
#from ACKmeans import *
from method import *
from data_loader import *
from functions import *
import torch.nn.functional as F

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    '''初始化超参'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoch", type=int, default=30, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2048, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--step_decay_weight", type=float, default=0.95, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lr_decay_step", type=float, default=10000, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--active_domain_loss_step", type=float, default=10000, help="")
    parser.add_argument("--model_root", type=str, default='model', help="")
    parser.add_argument("--cuda", type=bool, default=True, help="")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="")
    parser.add_argument("--alpha_weight", type=float, default=0.01, help="")
    parser.add_argument("--beta_weight", type=float, default=0.075, help="")
    parser.add_argument("--gamma_weight", type=float, default=0.25, help="")
    parser.add_argument("--betas", type=float, default=(0.9, 0.999), help="")
    parser.add_argument("--momentum", type=float, default=0.9, help="")
    parser.add_argument("--image_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--dataset_name", type=str, default="Youtube", help="name of the dataset")
    parser.add_argument("--channal_size", type=int, default=1000, help="")
    parser.add_argument("--path", type=str, default='self_feature_list.pt', help="")

    '''聚类超参'''
    parser.add_argument("--data_path", type=str, default=r'result', help="")
    parser.add_argument("--fin_epoch", type=int, default=50, help="training final epoch")
    parser.add_argument("--maxiter", type=int, default=1, help="")
    opt = parser.parse_args()

    '''设置随机种子'''
    manual_seed = random.randint(1, 10000)
    print('seed = ',manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    '''获取处理后的数据'''
#     data_path = './1000_1_ILSVRC2012_val_feature_file.npy'
    data_path = './2000_coco_feature_file.npy'
    training_data, num_classes, matdata_max_len = Self_data_process(data_path)  # 3,30
#     print(num_classes,matdata_max_len)
#     exit()
    D_training_data = training_data

    #     , num_classes, matdata_max_len = Self_data_process(data_path)  # 3,30

    dataloader = torch.utils.data.DataLoader(
        dataset=training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0)

    D_dataloader = torch.utils.data.DataLoader(
        dataset=D_training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0)
    print('111')
    '''导入模型'''
    Generator = SFG(input_size=matdata_max_len, output_size=matdata_max_len, c_size=opt.channal_size)  # 30,30
    Discriminator = SFD(input_size=matdata_max_len, output_size=num_classes, c_size=opt.channal_size)  # 30,3/1
    Encoder = En(input_size=matdata_max_len, output_size=matdata_max_len, c_size=opt.channal_size)  # 30,30
    Decoder = De(input_size=2*matdata_max_len, output_size=matdata_max_len, c_size=opt.channal_size)  # 60,30

    '''设置模型优化参数'''
#     G_optimizer = optim.SGD(Generator.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
#     D_optimizer = optim.SGD(Discriminator.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
#     En_optimizer = optim.SGD(Encoder.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
#     De_optimizer = optim.SGD(Decoder.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    G_optimizer = optim.Adam(Generator.parameters(), lr=opt.lr, betas=opt.betas, weight_decay=opt.weight_decay)
    D_optimizer = optim.Adam(Discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    En_optimizer = optim.Adam(Encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    De_optimizer = optim.Adam(Decoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    '''设置损失函数'''
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    adversarial_loss = torch.nn.BCELoss()
#     diff_loss = DiffLoss()
    diff_loss = MylossFunc()
#     recon_loss = torch.nn.KLDivLoss(reduction = 'batchmean')
    recon_loss = torch.nn.MSELoss()
    # loss_similarity = torch.nn.CrossEntropyLoss()

    if opt.cuda:
        Generator = Generator.cuda()
        Discriminator = Discriminator.cuda()
        Encoder = Encoder.cuda()
        Decoder = Decoder.cuda()

        auxiliary_loss = auxiliary_loss.cuda()
        adversarial_loss = adversarial_loss.cuda()
        diff_loss = diff_loss.cuda()
        recon_loss = recon_loss.cuda()

    for p in Generator.parameters() and Discriminator.parameters() and Encoder.parameters() and Decoder.parameters():
        p.requires_grad = True

    ''' '''
    Adv_loss_list = []
#     d_loss_list = []
    di_loss_list = []
    re_loss_list = []
    loss_list = []
    # show_config(opt)

    '''  training  '''
#     torch.backends.cudnn.enable =True
#     torch.backends.cudnn.benchmark = True
    FloatTensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if opt.cuda else torch.LongTensor

    valid = Variable(FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)

    for epoch in range(opt.n_epoch):
        for i in range(len(dataloader)):

            loss = 0

            train_data_imgs, train_data_label = next(iter(dataloader))
            D_data_imgs, D_data_label = next(iter(D_dataloader))

            train_data_img = train_data_imgs.type(torch.cuda.FloatTensor)
            D_data_img = D_data_imgs.type(torch.cuda.FloatTensor)

            #  Train Generator
            G_optimizer.zero_grad()
            G_data_fake = Generator(train_data_img)
            re_G_data_fake = ReverseLayerF.apply(G_data_fake)
            pred_data, pred_label = Discriminator(G_data_fake)
            re_pred_data, re_pred_label = Discriminator(re_G_data_fake)
#             g_loss = 0.5 * (adversarial_loss(pred_data, valid) + auxiliary_loss(re_pred_label,train_data_label.type(torch.cuda.LongTensor)))
#             g_loss.backward(retain_graph=True)
#             loss += g_loss
            g1_loss = 0.5 * (adversarial_loss(pred_data, valid))
            g2_loss = 0.5 * (auxiliary_loss(re_pred_label,train_data_label.type(torch.cuda.LongTensor)))
            g1_loss.backward(retain_graph=True)
            g2_loss.backward(retain_graph=True)
            loss += g1_loss
            loss += g2_loss

            #  Train Discriminator
            D_optimizer.zero_grad()
            real_pred, real_aux = Discriminator(D_data_img)
            fake_pred, fake_aux = Discriminator(G_data_fake.detach())
#             re_G_data_fake = ReverseLayerF.apply(G_data_fake)
#             re_fake_pred, re_fake_aux = Discriminator(re_G_data_fake.detach())

#             real_loss = 0.5*(adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, D_data_label.type(torch.cuda.LongTensor)))
#             fake_loss = 0.5*(adversarial_loss(fake_pred, fake)+ auxiliary_loss(re_fake_aux, train_data_label.type(torch.cuda.LongTensor)))
#             d_loss = (real_loss + fake_loss) / 2
#             d_loss.backward(retain_graph=True)
#             d_loss.backward(retain_graph=True)
#             loss += d_loss

            real_loss = 0.5*(adversarial_loss(real_pred, valid))
            fake_loss = 0.5*(adversarial_loss(fake_pred, fake))
            d1_loss = (real_loss + fake_loss) / 2
            d2_loss = auxiliary_loss(real_aux, D_data_label.type(torch.cuda.LongTensor))
            d1_loss.backward(retain_graph=True)
            d2_loss.backward(retain_graph=True)
            loss += d1_loss
            loss += d2_loss
                                     
            #  Train Encoder
            En_optimizer.zero_grad()
            re_train_data_img = ReverseLayerF.apply(train_data_img)
            prevate_feature = Encoder(re_train_data_img)

            di_loss = diff_loss(prevate_feature, G_data_fake.detach())   #detach()
            di_loss.backward(retain_graph=True)
            loss += di_loss

            #  Train Decoder
            De_optimizer.zero_grad()
            result_feature = Decoder(torch.cat((prevate_feature, G_data_fake),1))
#             re_loss = recon_loss(F.log_softmax(result_feature, dim=1), F.softmax(train_data_img, dim=1)) #result_feature.log(),train_data_img)

            re_loss = recon_loss(train_data_img, result_feature)
            re_loss.backward(retain_graph=True)
            loss += re_loss

#             g1_loss_list.append(float(g_loss))
#             g2_loss_list.append(float(g_loss))
#             d1_loss_list.append(float(d_loss))
#             d2_loss_list.append(float(d_loss))   
            Adv_loss_list.append((float(g1_loss)+float(g2_loss)+float(d1_loss)+float(d2_loss))/opt.batch_size)
            di_loss_list.append((float(di_loss))/opt.batch_size)
            re_loss_list.append((float(re_loss))/opt.batch_size)
            loss_list.append((float(loss))/opt.batch_size)

            # loss.backward(retain_graph=True)
            G_optimizer.step()
            D_optimizer.step()
            En_optimizer.step()
            De_optimizer.step()

            batches_done = epoch * len(D_dataloader) + i
#             print(
#             "[Batch %d/%d] [Adv loss: %f] [Diff loss: %f] [Recon loss: %f] [*loss: %f]"
#             % (i+1, len(dataloader), (d1_loss.item()+d2_loss.item()+g1_loss.item()+g2_loss.item())/opt.batch_size, di_loss.item()/opt.batch_size, re_loss.item()/opt.batch_size,loss.item()/opt.batch_size)
#         )
            
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f,%f] [G loss: %f,%f] [Diff loss: %f] [Recon loss: %f] [*loss: %f]"
            % (epoch+1, opt.n_epoch, i+1, len(dataloader), d1_loss.item(), d2_loss.item(), g1_loss.item(), g2_loss.item(), di_loss.item(), re_loss.item(),
               loss.item())
        )
#     f=open("L_Adv.txt","w")
#     f.write(str(Adv_loss_list))
#     f.close()
    
#     f=open("L_Con.txt","w")
#     f.write(str(re_loss_list))
#     f.close()
    
#     f=open("L_Diff.txt","w")
#     f.write(str(di_loss_list))
#     f.close()
    
#     f=open("T_loss.txt","w")
#     f.write(str(loss_list))
#     f.close()
    
    # plot_method(g_loss_list, d_loss_list, di_loss_list, re_loss_list, loss_list)
    # save_image(result_feature, 'pic/rec_image.png', nrow=16)
    torch.save(Generator.state_dict(), 'result' + '/Gen_epoch_' + str(epoch + 1) + '.pth')
    torch.save(Encoder.state_dict(), 'result' + '/En_epoch_' + str(epoch + 1) + '.pth')
    torch.save(Decoder.state_dict(), 'result' + '/De_epoch_' + str(epoch + 1) + '.pth')
    fin_Generator = os.path.join('result' + '/Gen_epoch_' + str(epoch + 1) + '.pth')
    fin_Encoder = os.path.join('result' + '/En_epoch_' + str(epoch + 1) + '.pth')
    fin_Decoder = os.path.join('result' + '/De_epoch_' + str(epoch + 1) + '.pth')

    '''  testing  '''
    # test(opt)
    #
    # '''  cluster  '''
    #Kmeans_cluster(opt)