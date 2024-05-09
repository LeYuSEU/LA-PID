import json
import os
import matplotlib.pyplot as plt
import numpy as np

json_path = r'/home/lhq/code/PID-FewShot/43.66+mini_imagenet_full_size+4-CONV+20w5s15q+PID+maml+bs2+minLR0.0001+metaLR0.001+epoches60+meanStd'

with open(os.path.join(json_path, 'logs', 'summary_statistics.json'), "r", encoding="utf-8") as f:
    content = json.load(f)
    train_loss = content['train_loss_mean']
    val_loss = content['val_loss_mean']
    train_acc = content['train_accuracy_mean']
    val_acc = content['val_accuracy_mean']

    num_len = len(train_loss)
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(num_len), train_loss, linewidth=3)
    plt.plot(np.arange(num_len), val_loss, linewidth=3)
    plt.legend(['train_loss', 'val_loss'])
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(num_len), train_acc, linewidth=3)
    plt.plot(np.arange(num_len), val_acc, linewidth=3)
    plt.legend(['train_acc', 'val_acc'])

    # plt.show()
    plt.savefig(os.path.join(json_path, 'curve.jpg'), dpi=600)
