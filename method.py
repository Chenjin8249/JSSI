import matplotlib.pyplot as plt

def show_config(config):
    print('        Training Begin ...')
    print('        数据集: ', config.dataset_name)
    print('        训练轮次: ',config.n_epoch)
    print('        图片尺寸: ', config.image_size,'*',config.image_size)
    print('        ')


def plot_method(g_loss_list, d_loss_list, di_loss_list, re_loss_list, loss_list):
    pic = plt.plot(g_loss_list,label='g_loss')
    plt.xlabel('Epoch')
    plt.ylabel('G_Loss')
    plt.legend()
    plt.show()
    pic = plt.plot(d_loss_list,label='d_loss')
    plt.xlabel('Epoch')
    plt.ylabel('D_Loss')
    plt.legend()
    plt.show()
    pic = plt.plot(di_loss_list,label='di_loss')
    plt.xlabel('Epoch')
    plt.ylabel('DI_Loss')
    plt.legend()
    plt.show()
    pic = plt.plot(re_loss_list,label='re_loss')
    plt.xlabel('Epoch')
    plt.ylabel('RE_Loss')
    plt.legend()
    plt.show()
    pic = plt.plot(loss_list,label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()      