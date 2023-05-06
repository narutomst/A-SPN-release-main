# 对A-SPN-release-main中model.py文件中的59行开始的spn()函数做一个修改，
# 使其符合最新的tensorflow keras神经网络定义形式
# 即将代码TensorFlow 1.x 迁移到 TensorFlow 2.x

def spn(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')

    F = Reshape([img_rows * img_cols, num_PC])(CNNInput)
    F = BatchNormalization()(F)
    F = Dropout(rate=0.5)(F)

    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='f2')(F)
    F = SecondOrderPooling(name='feature1')(F)
    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='feature2')(F)

    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=num_PC, c=nb_classes))(
        F)
    model = Model(inputs=[CNNInput], outputs=F)

    return model
	
