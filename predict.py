import time
from libraries import *
from loss_fns import dice_loss_fn, wce_loss_fn, calc_loss_weights
from model import SSA_Net
from dataset_loader import DatasetManager

class Predicter():
    
    def __init__(self, model:keras.Model=None, weights_path:str=None, n_classes:int=1):
        self.model = model
        self.weights_path = weights_path
        self.n_classes = n_classes
        
    def __load_model(self, model, weights_path):
        # Default optimizer
        model.build((None, 512, 512, 2))
        
        model.load_weights(weights_path)
        return model
            
    # i couldn't upload weights so i comment it out:)
    # def predict_single_class_1(self):
    #     ssa_net = SSA_Net(class_nums=1)
    #     ssa_net = self.__load_model(ssa_net, '/content/Weights/Single class/')
    #     ct_slice = tf.expand_dims(self.__normalize(nib.load('/content/Data/Dataset 1/CT scans/coronacases_001.nii').get_fdata()[:, :, 117]), axis=2)
    #     lung_mask = tf.expand_dims(nib.load('/content/Data/Dataset 1/Lung mask/coronacases_001.nii').get_fdata()[:, :, 117].astype("float32"), axis=2)
    #     input = tf.concat([ct_slice, lung_mask], 2)
    #     image = ssa_net.predict(tf.expand_dims(input, axis=0), 1)[0]
    #     infection_mask = tf.expand_dims(nib.load('/content/Data/Dataset 1/Infection mask/coronacases_001.nii').get_fdata()[:, :, 117].astype("float32"), axis=2)
    #     print(infection_mask[:, :, 0].shape, image[0, :, :, 0].shape, type(infection_mask[:, :, 0].numpy()), type(image[0, :, :, 0]))
    #     im = Image.fromarray(((infection_mask[:, :, 0].numpy() * 255)).astype(np.uint8))
    #     im.save("infection mask.png")
    #     im.show()
    #     im = Image.fromarray(((image[0, :, :, 0] * 255)).astype(np.uint8))
    #     im.save("predicted infection mask.png")
    #     im.show()
        
    def predict_ss_fs(self):
        data_manager = DatasetManager()
        data = data_manager.load_dataset_3()
        train_data, val_data, test_data = data_manager.split_data([data[0], data[1]], 'ssfs')
        ssa_net = SSA_Net(class_nums=2)
        ssa_net = self.__load_model(ssa_net, '/content/Weights/Multiclass(1 sample)/Epoch560.h5')
        input = test_data[0][0]
        image = ssa_net.predict(tf.expand_dims(input, axis=0), 1)[0]
        im = Image.fromarray(((test_data[1][0, :, :, 1] * 255)).astype(np.uint8))
        im.save("/content/infection mask 1.png")
        im.show()
        im = Image.fromarray(((test_data[1][0, :, :, 2] * 255)).astype(np.uint8))
        im.save("/content/infection mask 2.png")
        im.show()
        im = Image.fromarray(((image[0, :, :, 1] * 255)).astype(np.uint8))
        im.save("/content/predicted infection mask 1.png")
        im.show()
        im = Image.fromarray(((image[0, :, :, 2] * 255)).astype(np.uint8))
        im.save("/content/predicted infection mask 2.png")
        im.show()