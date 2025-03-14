
import numpy as np
from matplotlib import pyplot as plt
from sklearn import manifold

if __name__ == '__main__':
    #我这里是多个model一起画的，一个model的可视化要放一起算
    path_list = ['base-C0.1model.h5','base-C0.01model.h5', 'base-C0.7model.h5']
    for k in range(len(path_list)):
        fact = np.load('./tsne/' + path_list[k] + 'val.npy')#fact是特征向量，根据你的需求选择合适的网络中间过程输入，但别选最终输出
        y = np.load('./tsne/y_short.npy')#y是标签集，要求和特征对应，以便在可视化中标记点的实际类别
        #TSNE本质上是一个从高维到极低维的无监督数据降维算法
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=10)#定义TSNE过程
        X_tsne = tsne.fit_transform(fact)#进行TSNE变换
        # 归一化是有必要做的
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)
        #后面是画图代码
        colors = ['black',
                  plt.cm.tab20b(0), plt.cm.tab20b(1), plt.cm.tab20b(2), plt.cm.tab20b(3),
                  plt.cm.tab20b(4), plt.cm.tab20b(5), plt.cm.tab20b(6), plt.cm.tab20b(7),
                  plt.cm.tab20b(8), plt.cm.tab20b(9), plt.cm.tab20b(10), plt.cm.tab20b(11),
                  plt.cm.tab20b(12), plt.cm.tab20b(13), plt.cm.tab20b(14), plt.cm.tab20b(15),
                  plt.cm.tab20b(16), plt.cm.tab20b(17), plt.cm.tab20b(18), plt.cm.tab20b(19),
                  ]
        plt.figure(k, figsize=(12, 10))
        #这个reshape是因为我每一类的样本个数是100，TSNE结果是二维，正好可以这样简单的把不同类样本分开
        reshaped_matrix = X_norm.reshape(-1, 100, 2)
        #找到各类的中心，以便打数字标签
        centers = np.mean(reshaped_matrix, axis=1)
        for j in range(centers.shape[0]):
            #在类中心（偏一点的位置免得挡住）写类标签
            plt.text(centers[j, 0]+0.03, centers[j, 1]+0.03, str(j), color=colors[j], fontdict={'weight': 'bold', 'size': 12})
        for i in range(X_norm.shape[0]):
            #描点，标签i对应颜色i
            plt.plot(X_norm[i, 0], X_norm[i, 1], 'o', color=colors[y[i]])
        #画颜色柱的代码，有一个标签有点歪的bug
        cmap = plt.cm.colors.ListedColormap(colors)
        norm = plt.cm.colors.Normalize(vmin=0, vmax=len(colors) - 1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(sm, ticks=range(len(colors)), label='Fault Type')
        plt.show()
        plt.savefig('./tsne/snip' + str(k) + '.png', dpi=600)
