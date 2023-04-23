"""
SSA-Net Model
"""
from libraries import *

# encodder building block
# https://github.com/safwankdb/ResNet34-TF2/blob/master/model.py
# https://github.com/raghakot/keras-resnet/blob/master/resnet.py
class EncoderBlock(Model):
    def __init__(self, channels:int, name:str, stride:int=1):
        super(EncoderBlock, self).__init__(name = name)
        self.flag = (stride != 1)
        self.conv1 = Conv2D(channels, 3, stride, kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4), padding='same', name='_conv1')
        self.bn1 = BatchNormalization(name='_bn1')
        self.conv2 = Conv2D(channels, 3, kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4), padding='same', name='_conv2')
        self.bn2 = BatchNormalization(name='_bn2')
        self.relu = ReLU(name='_relu')
        if self.flag:
            self.bn3 = BatchNormalization(name='_bn3')
            self.conv3 = Conv2D(channels, 1, stride, kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4), name='_conv3')

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        if self.flag:
            x = self.conv3(x)
            x = self.bn3(x)
        x1 = Layers.add([x, x1])
        x1 = self.relu(x1)
        return x1
    
# decoder block
class DecoderBlock(Model):
    def __init__(self,  name_prefix:str, channel_1:int, channel_2:int, channel_3:int, name:str):
        super(DecoderBlock, self).__init__(name=name)
        self.name_prefix = name_prefix
        self.conv1 = Conv2D(channel_1, 1, padding='same', name=name_prefix+'_conv1')
        self.deconv1 = Conv2DTranspose(channel_2, 3, 2, 'same', name=name_prefix+'_deconv1')
        self.conv2 = Conv2D(channel_3, 1, padding='same', name=name_prefix+'_conv2')

    def call(self, x):
        x = self.conv1(x)
        x = self.deconv1(x)
        x = self.conv2(x)
        return x

class SSA_Net(Model):
    def __init__(self, class_nums:int=1, has_sc:bool=True, has_sa:bool=True):
        super(SSA_Net, self).__init__(name='SSA_Net')
        
        # train data specs
        self.class_nums = class_nums
        self.train_img_height = 512
        self.train_img_width = 512
        self.dims = (self.train_img_height, self.train_img_width, 2)
        
        # model specs
        self.sa = has_sa
        self.scn = has_sc
        
        # encoder layers
        # before first block
        self.concat = Concatenate(-1, name='concat') #?
        self.conv1 = Conv2D(64, 7, 2, 'same', name='conv1')# ? 
        self.mp1 = MaxPooling2D(3, 2, padding='same', name='mp1') #
        self.bn = BatchNormalization(name='bn1')
        self.relu = ReLU(name='relu1')
        
        # block 1
        self.conv2_1 = EncoderBlock(64, name='en2_1')
        self.conv2_2 = EncoderBlock(64, name='en2_2')
        self.conv2_3 = EncoderBlock(64, name='en2_3')
        
        # block 2
        self.conv3_1 = EncoderBlock(128, stride=2, name='en3_1')
        self.conv3_2 = EncoderBlock(128, name='en3_2')
        self.conv3_3 = EncoderBlock(128, name='en3_3')
        self.conv3_4 = EncoderBlock(128, name='en3_4')
        
        # block 3
        self.conv4_1 = EncoderBlock(256, stride=2, name='en4_1')
        self.conv4_2 = EncoderBlock(256, name='en4_2')
        self.conv4_3 = EncoderBlock(256, name='en4_3')
        self.conv4_4 = EncoderBlock(256, name='en4_4')
        self.conv4_5 = EncoderBlock(256, name='en4_5')
        self.conv4_6 = EncoderBlock(256, name='en4_6')
        
        # block 4
        self.conv5_1 = EncoderBlock(512, stride=2, name='en5_1')
        self.conv5_2 = EncoderBlock(512, name='en5_2')
        self.conv5_3 = EncoderBlock(512, name='en5_3')
        
        # decoder layers
        # block 1
        self.de1 = DecoderBlock('de1', 512, 256, 256, name='de1')
        # block 2
        self.de2 = DecoderBlock('de2', 256, 128, 128, name='de2')
        self.de2_concat = Concatenate(-1, name='de2_concat')
        # block 3
        self.de3 = DecoderBlock('de3', 128, 64, 64, name='de3')
        self.de3_concat = Concatenate(-1, name='de3_concat')
        # block 4
        self.de4 = DecoderBlock('de4', 64, 64, 64, name='de4')
        self.de4_concat = Concatenate(-1, name='de4_concat')
        # block 5
        self.de5 = DecoderBlock('de5', 64, 64, 64, name='de5')
        self.de5_concat = Concatenate(-1, name='de5_concat')
        # final output sigmoid layer
        if self.class_nums != 1:
            self.output_layer = Conv2D(self.class_nums + 1, 1, activation='sigmoid', name='output_layer')
        else:
            self.output_layer = Conv2D(1, 1, activation='sigmoid', name='output_layer')
        
    # model default call method
    def call(self, inputs):
        
        # get inputs
        # ct_slice = inputs[0]
        # ct_mask = inputs[1]
        x = inputs
        
        # before feeding data to encoder
        x0 = self.relu(self.bn(self.conv1(x)))
        
        # encoder block 1
        x1 = self.mp1(x0)
        x1 = self.conv2_1(x1)
        x1 = self.conv2_2(x1)
        x1 = self.conv2_3(x1)
        
        # encoder block 2
        x2 = self.conv3_1(x1)
        x2 = self.conv3_2(x2)
        x2 = self.conv3_3(x2)
        x2 = self.conv3_4(x2)
        
        # encoder block 3
        x3 = self.conv4_1(x2)
        x3 = self.conv4_2(x3)
        x3 = self.conv4_3(x3)
        x3 = self.conv4_4(x3)
        x3 = self.conv4_5(x3)
        x3 = self.conv4_6(x3)
        
        # encoder block 4
        x4 = self.conv5_1(x3)
        x4 = self.conv5_2(x4)
        x4 = self.conv5_3(x4)
        
        # spatial convolution
        # https://github.com/cardwing/Codes-for-Lane-Detection/tree/master/SCNN-Tensorflow
        if self.scn == True:
            # top to down
            feature_list_old = []
            feature_list_new = []
            n_dims_1 = x4.get_shape().as_list()[1]
            for cnt in range(n_dims_1):
                feature_list_old.append(tf.expand_dims(x4[:, cnt, :, :], axis=1))
            feature_list_new.append(tf.expand_dims(x4[:, 0, :, :], axis=1))
            
            # code with keras
            with tf.compat.v1.variable_scope("x4s_td"):
                x4_td = Add()([ReLU()(Conv2D(512, 1, 1, kernel_initializer='random_normal')(feature_list_old[0])),
                                    feature_list_old[1]])
                feature_list_new.append(x4_td)
            for cnt in range(2, n_dims_1):
                with tf.compat.v1.variable_scope("x4s_td", reuse=True):
                    x4_td = Add()([ReLU()(Conv2D(512, 1, 1, kernel_initializer='random_normal')(feature_list_new[cnt - 1])),
                                        feature_list_old[cnt]])
                    feature_list_new.append(x4_td)
            
            #down to top
            feature_list_old = feature_list_new
            feature_list_new = []
            length = int(self.train_img_height / 32) - 1 # ? why not the last one
            feature_list_new.append(feature_list_old[length])
            
            # code with keras
            with tf.compat.v1.variable_scope("x4s_dt"):
                x4_dt = Add()([ReLU()(Conv2D(512, 1, 1, kernel_initializer='random_normal')(feature_list_old[length])),
                                    feature_list_old[length - 1]])
                feature_list_new.append(x4_dt)

            for cnt in range(2, n_dims_1):
                with tf.compat.v1.variable_scope("x4s_dt", reuse=True):
                    x4_dt = Add()([ReLU()(Conv2D(512, 1, 1, kernel_initializer='random_normal')(feature_list_new[cnt - 1])),
                                        feature_list_old[length - cnt]])
                    feature_list_new.append(x4_dt)
                    
            # reverse the list to get order identical to input image
            feature_list_new.reverse()
            # stack feature lists to get next part input
            processed_feature = tf.stack(feature_list_new, axis=1)
            processed_feature = tf.squeeze(processed_feature, axis=2)
            
            # left to right
            feature_list_old = []
            feature_list_new = []
            n_dims_2 = processed_feature.get_shape().as_list()[2]
            for cnt in range(n_dims_2):
                feature_list_old.append(tf.expand_dims(processed_feature[:, :, cnt, :], axis=2))
            feature_list_new.append(tf.expand_dims(processed_feature[:, :, 0, :], axis=2))
            
            # code with keras
            with tf.compat.v1.variable_scope("x4s_lr"):
                x4_lr = Add()([ReLU()(Conv2D(512, 1, 1, kernel_initializer='random_normal')(feature_list_old[0])),
                                    feature_list_old[1]])
                feature_list_new.append(x4_lr)

            for cnt in range(2, n_dims_2):
                with tf.compat.v1.variable_scope("x4s_lr", reuse=True):
                    x4_lr = Add()([ReLU()(Conv2D(512, 1, 1, kernel_initializer='random_normal')(feature_list_new[cnt - 1])),
                                        feature_list_old[cnt]])
                    feature_list_new.append(x4_lr)
                    
            # right to left
            feature_list_old = feature_list_new
            feature_list_new = []
            length = int(self.train_img_width / 32) - 1
            feature_list_new.append(feature_list_old[length])
            
            # code with keras
            with tf.compat.v1.variable_scope("x4s_rl"):
                x4_rl = Add()([ReLU()(Conv2D(512, 1, 1, kernel_initializer='random_normal')(feature_list_old[length])),
                                    feature_list_old[length - 1]])
                feature_list_new.append(x4_rl)

            for cnt in range(2, processed_feature.get_shape().as_list()[2]):
                with tf.compat.v1.variable_scope("x4s_rl", reuse=True):
                    x4_rl = Add()([ReLU()(Conv2D(512, 1, 1, kernel_initializer='random_normal')(feature_list_new[cnt - 1])),
                                        feature_list_old[length - cnt]])
                    feature_list_new.append(x4_rl)
                    
            # reverse the list to get order identical to input image
            feature_list_new.reverse()
            
            # stack layers to have input for decoder part
            processed_feature = tf.stack(feature_list_new, axis=2)
            processed_feature = tf.squeeze(processed_feature, axis=3)
            x5 = processed_feature
        
        # decoder block 1
        if self.scn == True:
            x6 = self.de1(x5)
        else:
            x6 = self.de1(x4)
        
        # decoder block 2
        x7 = self.de2_concat([x6, x3])
        x7 = self.de2(x7)
        
        # decoder block 3
        x8 = self.de3_concat([x7, x2])
        x8 = self.de3(x8)
        
        # decoder block 4
        x9 = self.de4_concat([x8, x1])
        x9 = self.de4(x9)
        
        # decoder block 5
        x10 = self.de5_concat([x9, x0])
        x10 = self.de5(x10)
        
        # final sigmoid layer
        output = self.output_layer(x10)
        
        # return outputs (final mask, four decoder outputs for feeding to loss function)
        if self.sa == True:
            return [output, x1, x2, x3, x4]
        else:
            return [output]
        
    # custom build_graph method for building model graph
    # to print model's summary and plot model like sequential model
    def build_graph(self):
        x = Input(shape=self.dims)
        return Model(inputs=[x], outputs=self.call(x), name='SSA_Net')