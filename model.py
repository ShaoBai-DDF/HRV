# 保存模型/加载模型
from keras.models import load_model
import matplotlib.pyplot as plt


# model.save('model_p01')  # creates a HDF5 file 'dl_T.h5'
# del model  # 删除现有模型
# # 加载已保存模型
model = load_model('model_p01')
acc = h.history['accuracy']
val_acc = h.history['val_accuracy']
loss = h.history['loss']
val_loss = h.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs,acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
