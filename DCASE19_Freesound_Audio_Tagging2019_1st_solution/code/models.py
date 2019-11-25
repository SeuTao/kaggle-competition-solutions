from keras.layers import *
from time_frequency import Melspectrogram, AdditiveNoise
from keras.optimizers import Nadam,SGD
from keras.constraints import *
from keras.initializers import *
from keras.models import Model
from config import *

EPS = 1e-8

def squeeze_excitation_layer(x, out_dim, ratio = 4):
    '''
    SE module performs inter-channel weighting.
    '''
    squeeze = GlobalAveragePooling2D()(x)
    excitation = Dense(units=out_dim // ratio)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, out_dim))(excitation)

    scale = multiply([x, excitation])
    return scale

def conv_se_block(x,filters,pool_stride,pool_size,pool_mode,cfg, ratio = 4):

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)
    x = squeeze_excitation_layer(x, out_dim=filters,ratio=ratio)
    x = pooling_block(x, pool_size[0], pool_stride[0], pool_mode[0], cfg)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)
    x = squeeze_excitation_layer(x, out_dim=filters,ratio=ratio)
    x = pooling_block(x, pool_size[1], pool_stride[1], pool_mode[1], cfg)

    return x

def AveMaxPool(x, pool_size,stride, ave_axis):
    if isinstance(pool_size,int):
        pool_size1,pool_size2 = pool_size, pool_size
    else:
        pool_size1,pool_size2 = pool_size
    if ave_axis == 2:
        x = AveragePooling2D(pool_size=(1,pool_size1), padding='same', strides=(1,stride))(x)
        x = MaxPool2D(pool_size=(pool_size2,1), padding='same', strides=(stride,1))(x)
    elif ave_axis == 1:
        x = AveragePooling2D(pool_size=(pool_size1,1), padding='same', strides=(stride,1))(x)
        x = MaxPool2D(pool_size=(1,pool_size2), padding='same', strides=(1,stride))(x)
    elif ave_axis == 3:
        x = MaxPool2D(pool_size=(1,pool_size1), padding='same', strides=(1,stride))(x)
        x = AveragePooling2D(pool_size=(pool_size2, 1), padding='same', strides=(stride, 1))(x)
    elif ave_axis == 4:
        x = MaxPool2D(pool_size=(pool_size1, 1), padding='same', strides=(stride, 1))(x)
        x = AveragePooling2D(pool_size=(1, pool_size2), padding='same', strides=(1, stride))(x)
    else:
        raise RuntimeError("axis error")
    return x

def pooling_block(x,pool_size,stride,pool_mode, cfg):
    if pool_mode == 'max':
        x = MaxPool2D(pool_size=pool_size, padding='same', strides=stride)(x)
    elif pool_mode == 'ave':
        x = AveragePooling2D(pool_size=pool_size, padding='same', strides=stride)(x)
    elif pool_mode == 'avemax1':
        x = AveMaxPool(x, pool_size=pool_size, stride=stride, ave_axis=1)
    elif pool_mode == 'avemax2':
        x = AveMaxPool(x, pool_size=pool_size, stride=stride, ave_axis=2)
    elif pool_mode == 'avemax3':
        x = AveMaxPool(x, pool_size=pool_size, stride=stride, ave_axis=3)
    elif pool_mode == 'avemax4':
        x = AveMaxPool(x, pool_size=pool_size, stride=stride, ave_axis=4)
    elif pool_mode == 'conv':
        x = Lambda(lambda x:K.expand_dims(K.permute_dimensions(x,(0,3,1,2)),axis=-1))(x)
        x = TimeDistributed(Conv2D(filters=1, kernel_size=pool_size, strides=stride, padding='same', use_bias=False))(x)
        x = Lambda(lambda x:K.permute_dimensions(K.squeeze(x,axis=-1),(0,2,3,1)))(x)
    elif pool_mode is None:
        x = x
    else:
        raise RuntimeError('pool mode error')
    return x

def conv_block(x,filters,pool_stride,pool_size,pool_mode,cfg):


    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)
    x = pooling_block(x, pool_size[0], pool_stride[0], pool_mode[0], cfg)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)
    x = pooling_block(x, pool_size[1], pool_stride[1], pool_mode[1], cfg)
    return x


def conv_cat_block(x, filters, pool_stride, pool_size, pool_mode, cfg):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)
    x = pooling_block(x, pool_size[0], pool_stride[0], pool_mode[0], cfg)

    x1 = x
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)

    ## concat
    x = concatenate([x1, x])
    x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = pooling_block(x, pool_size[1], pool_stride[1], pool_mode[1], cfg)

    return x


def conv_se_cat_block(x, filters, pool_stride, pool_size, pool_mode, cfg):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)
    x = squeeze_excitation_layer(x, out_dim=filters, ratio=4)
    x = pooling_block(x, pool_size[0], pool_stride[0], pool_mode[0], cfg)
    x1 = x
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)
    x = squeeze_excitation_layer(x, out_dim=filters, ratio=4)
    ## concat
    x = concatenate([x1, x])
    x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = pooling_block(x, pool_size[1], pool_stride[1], pool_mode[1], cfg)

    return x



def pixelShuffle(x):
    _,h,w,c = K.int_shape(x)
    bs = K.shape(x)[0]
    assert w%2==0
    x = K.reshape(x,(bs,h,w//2,c*2))

    # assert h % 2 == 0
    # x = K.permute_dimensions(x,(0,2,1,3))
    # x = K.reshape(x,(bs,w//2,h//2,c*4))
    # x = K.permute_dimensions(x,(0,2,1,3))
    return x

def get_se_backbone(x, cfg):

    x = Conv2D(64, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)
    x = squeeze_excitation_layer(x, out_dim=64, ratio=4)
    # backbone
    x = conv_se_block(x, 96, (1, 2), (3, 2), cfg.pool_mode, cfg)
    x = conv_se_block(x, 128, (1, 2), (3, 2), cfg.pool_mode, cfg)
    x = conv_se_block(x, 256, (1, 2), (3, 3), cfg.pool_mode, cfg)
    x = conv_se_block(x, 512, (1, 2), (3, 2), (None, None), cfg)  ## [bs,  54, 8, 512]

    # global pooling
    x = Lambda(pixelShuffle)(x)  ## [bs,  54, 4, 1024]
    x = Lambda(lambda x: K.max(x, axis=1))(x)
    x = Lambda(lambda x: K.mean(x, axis=1))(x)

    return x

def get_conv_backbone(x, cfg):

    # input stem
    x = Conv2D(64, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)

    # backbone
    x = conv_block(x, 96, (1, 2), (3, 2), cfg.pool_mode, cfg)
    x = conv_block(x, 128, (1, 2), (3, 2), cfg.pool_mode, cfg)
    x = conv_block(x, 256, (1, 2), (3, 3), cfg.pool_mode, cfg)
    x = conv_block(x, 512, (1, 2), (3, 2), (None, None), cfg)  ## [bs,  54, 8, 512]

    # global pooling
    x = Lambda(pixelShuffle)(x)  ## [bs,  54, 4, 1024]
    x = Lambda(lambda x: K.max(x, axis=1))(x)
    x = Lambda(lambda x: K.mean(x, axis=1))(x)

    return x

def get_se_cat_backbone(x,cfg):


    x = Conv2D(64, kernel_size=3, padding='same',use_bias=False)(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)
    x = squeeze_excitation_layer(x, out_dim=64,ratio=4)
    # backbone
    x = conv_se_cat_block(x, 96, (1,2), (3,2), cfg.pool_mode, cfg)
    x = conv_se_cat_block(x, 128, (1,2), (3,2), cfg.pool_mode, cfg)
    x = conv_se_cat_block(x, 256, (1,2), (3,3), cfg.pool_mode, cfg)
    x = conv_se_cat_block(x, 512, (1,2), (3,2), (None,None), cfg) ## [bs,  54, 8, 512]

    # global pooling
    x = Lambda(pixelShuffle)(x)  ## [bs,  54, 4, 1024]
    x = Lambda(lambda x: K.max(x, axis=1))(x)
    x = Lambda(lambda x: K.mean(x, axis=1))(x)
    return x

def get_concat_backbone(x, cfg):

    # input stem
    x = Conv2D(64, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)

    # backbone
    x = conv_cat_block(x, 96, (1, 2), (3, 2), cfg.pool_mode, cfg)
    x = conv_cat_block(x, 128, (1, 2), (3, 2), cfg.pool_mode, cfg)
    x = conv_cat_block(x, 256, (1, 2), (3, 3), cfg.pool_mode, cfg)
    x = conv_cat_block(x, 512, (1, 2), (3, 2), (None, None), cfg)  ## [bs,  54, 8, 512]

    # global pooling
    x = Lambda(pixelShuffle)(x)  ## [bs,  54, 4, 1024]
    x = Lambda(lambda x: K.max(x, axis=1))(x)
    x = Lambda(lambda x: K.mean(x, axis=1))(x)

    return x

def model_se_MSC(x, cfg):
    ratio = 4
    # input stem
    x_3 = Conv2D(32, kernel_size=3, padding='same', use_bias=False)(x)
    x_5 = Conv2D(32, kernel_size=5, padding='same', use_bias=False)(x)
    x_7 = Conv2D(32, kernel_size=7, padding='same', use_bias=False)(x)

    x = concatenate([x_3, x_5, x_7])
    x = BatchNormalization(momentum=cfg.momentum)(x)
    x = Activation('relu')(x)
    x = squeeze_excitation_layer(x, out_dim=96, ratio=ratio)

    w_ratio = cfg.w_ratio
    # backbone
    x = conv_se_block(x, int(96 * w_ratio), (1, 2), (3, 2), cfg.pool_mode, cfg, ratio=ratio)
    x = conv_se_block(x, int(128 * w_ratio), (1, 2), (3, 2), cfg.pool_mode, cfg, ratio=ratio)
    x = conv_se_block(x, int(256 * w_ratio), (1, 2), (3, 3), cfg.pool_mode, cfg, ratio=ratio)
    x = conv_se_block(x, int(512 * w_ratio), (1, 2), (3, 2), (None, None), cfg, ratio=ratio)

    # global pooling
    x = Lambda(pixelShuffle)(x)
    x = Lambda(lambda x: K.max(x, axis=1))(x)
    x = Lambda(lambda x: K.mean(x, axis=1))(x)
    return x

def cnn_model(cfg):


    x_in = Input((cfg.maxlen,), name='audio')
    feat_in = Input((1,), name='other')
    feat = feat_in

    gfeat_in = Input((128, 12), name='global_feat')
    gfeat = BatchNormalization()(gfeat_in)
    gfeat = Bidirectional(CuDNNGRU(cfg.rnn_unit, return_sequences=True), merge_mode='sum')(gfeat)
    gfeat = Bidirectional(CuDNNGRU(cfg.rnn_unit, return_sequences=True), merge_mode='sum')(gfeat)
    gfeat = GlobalMaxPooling1D()(gfeat)

    x = Lambda(lambda t: K.expand_dims(t, axis=1))(x_in)
    x_mel = Melspectrogram(n_dft=1024, n_hop=cfg.stride, input_shape=(1, K.int_shape(x_in)[1]),
                           # n_hop -> stride   n_dft kernel_size
                           padding='same', sr=44100, n_mels=cfg.n_mels,
                           power_melgram=cfg.pm, return_decibel_melgram=True,
                           trainable_fb=False, trainable_kernel=False,
                           image_data_format='channels_last', trainable=False)(x)

    x_mel = Lambda(lambda x: K.permute_dimensions(x, pattern=(0, 2, 1, 3)))(x_mel)
    x = cfg.get_backbone(x_mel, cfg)
    x = concatenate([x, gfeat, feat])
    output = Dense(units=n_classes, activation='sigmoid')(x)

    y_in = Input((n_classes,), name='y')
    y = y_in

    def get_loss(x):
        y_true, y_pred = x
        loss1 = K.mean(K.binary_crossentropy(y_true, y_pred))
        return loss1

    loss = Lambda(get_loss)([y, output])
    model = Model(inputs=[x_in, feat_in, gfeat_in, y_in], outputs=[output])

    if cfg.pretrained is not None:
        model.load_weights("../model/{}.h5".format(cfg.pretrained))
        print('load_pretrained_success...')

    model.add_loss(loss)
    model.compile(
        # loss=get_loss,
        optimizer=Nadam(lr=cfg.lr),
    )
    return model


class normNorm(Constraint):
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        # w = K.relu(w)
        # w = K.clip(w,-0.5,1)
        w /= (K.sum(w**2, axis=self.axis, keepdims=True)**0.5)
        return w

    def get_config(self):
        return {'axis': self.axis}

def stacker(cfg,n):
    def kinit(shape, name=None):
        value = np.zeros(shape)
        value[:, -1] = 1
        return K.variable(value, name=name)


    x_in = Input((80,n))
    x = x_in
    # x = Lambda(lambda x: 1.5*x)(x)
    x = LocallyConnected1D(1,1,kernel_initializer=kinit,kernel_constraint=normNorm(1),use_bias=False)(x)
    x = Flatten()(x)
    x = Dense(80, use_bias=False, kernel_initializer=Identity(1))(x)
    x = Lambda(lambda x: (x - 1.6))(x)
    x = Activation('tanh')(x)
    x = Lambda(lambda x:(x+1)*0.5)(x)

    model = Model(inputs=x_in, outputs=x)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Nadam(lr=cfg.lr),
    )
    return model


if __name__ == '__main__':
    cfg = Config()
    model = cnn_model(cfg)
    print(model.summary())





