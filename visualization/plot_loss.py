import numpy as np
import seaborn as sns

file = 'loss_synthetic_images.npz'
a = np.load(file)

num = 20
loss1 = a['loss1']
loss2 = a['loss2']

def plot(loss1, loss2):
    sns.set(font_scale=1.5)

    data = {
        'epoch': list(range(len(loss1[::num]))) + list(range(len(loss2[::num]))),
        'loss': list(loss1[::num]) + list(loss2[::num]),
        'model': ['Base'] * len(loss1[::num]) + ['CVAE'] * len(loss2[::num]),
    }
    printout = {k: len(v) for k, v in data.items() }
    print('check length:', printout)
    plot = sns.lineplot(data=data, x='epoch', y='loss', hue="model",
                 )
    plot.set(ylim=(0, 0.4), xlabel='Epoch', ylabel="Reconstruction Error",)

    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.show()

plot(loss1, loss2)

