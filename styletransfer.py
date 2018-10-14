from keras import backend as K
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

contentImPath = './data/jade.jpg'
styleImPath = './data/anime2.jpg'

gimgout = './output.jpg'

targetHeight = 512
targetWidth = 512

cImageOrig = Image.open(contentImPath)
cImageSizeOrig = cImageOrig.size

targetSize = (targetHeight, targetWidth)

cImage = load_img(path=contentImPath, target_size=targetSize)
cImageArray = img_to_array(cImage)
cImageArray = K.variable(preprocess_input(np.expand_dims(cImageArray, axis=0)), dtype='float32')

sImage = load_img(path=styleImPath, target_size=targetSize)
sImageArray = img_to_array(sImage)
sImageArray = K.variable(preprocess_input(np.expand_dims(sImageArray, axis=0)), dtype='float32')

rimg = np.random.randint(256, size=(targetWidth, targetHeight, 3)).astype('float64')
rimg = preprocess_input(np.expand_dims(rimg, axis=0))
rimgplaceholder = K.placeholder(shape=(1, targetWidth, targetHeight, 3))

def get_feature_reps(x, layer_names, model):
    featMatrices = []
    for ln in layer_names:
        selectedLayer = model.get_layer(ln)
        featRaw = selectedLayer.output
        featRawShape = K.shape(featRaw).eval(session=tf_session)
        N_l = featRawShape[-1]
        M_l = featRawShape[1]*featRawShape[2]
        featMatrix = K.reshape(featRaw, (M_l, N_l))
        featMatrix = K.transpose(featMatrix)
        featMatrices.append(featMatrix)
    return featMatrices


def get_content_loss(F, P):
    cLoss = 0.5*K.sum(K.square(F - P))
    return cLoss

def get_Gram_matrix(F):
    G = K.dot(F, K.transpose(F))
    return G

def get_style_loss(ws, Gs, As):
    sLoss = K.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_l = K.int_shape(G)[1]
        N_l = K.int_shape(G)[0]
        G_gram = get_Gram_matrix(G)
        A_gram = get_Gram_matrix(A)
        sLoss+= w*0.25*K.sum(K.square(G_gram - A_gram))/ (N_l**2 * M_l**2)
    return sLoss

def get_total_loss(gImPlaceholder, alpha=1.0, beta=10000.0):
    F = get_feature_reps(gImPlaceholder, layer_names=[cLayerName], model=gModel)[0]
    Gs = get_feature_reps(gImPlaceholder, layer_names=sLayerNames, model=gModel)
    contentLoss = get_content_loss(F, P)
    styleLoss = get_style_loss(ws, Gs, As)
    totalLoss = alpha*contentLoss + beta*styleLoss
    return totalLoss

def calculate_loss(gImArr):
    """
    Calculate total loss using K.function
    """
    if gImArr.shape != (1, targetWidth, targetWidth, 3):
        gImArr = gImArr.reshape((1, targetWidth, targetHeight, 3))
    loss_fcn = K.function([gModel.input], [get_total_loss(gModel.input)])
    return loss_fcn([gImArr])[0].astype('float64')

def get_grad(gImArr):
    """
    Calculate the gradient of the loss function with respect to the generated image
    """
    if gImArr.shape != (1, targetWidth, targetHeight, 3):
        gImArr = gImArr.reshape((1, targetWidth, targetHeight, 3))
    grad_fcn = K.function([gModel.input], K.gradients(get_total_loss(gModel.input), [gModel.input]))
    grad = grad_fcn([gImArr])[0].flatten().astype('float64')
    return grad

def postprocess_array(x):
    # Zero-center by mean pixel
    if x.shape != (targetWidth, targetHeight, 3):
        x = x.reshape((targetWidth, targetHeight, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

def reprocess_array(x):
    x = np.expand_dims(x.astype('float64'), axis=0)
    x = preprocess_input(x)
    return x


def save_original_size(x, target_size=cImageSizeOrig, index=""):
    xIm = Image.fromarray(x)
    xIm = xIm.resize(target_size)
    xIm.save(gimgout+str(index)+'.jpg')
    return xIm


from keras.applications import VGG16
from scipy.optimize import fmin_l_bfgs_b

tf_session = K.get_session()
cModel = VGG16(include_top=False, weights='imagenet', input_tensor=cImageArray)
sModel = VGG16(include_top=False, weights='imagenet', input_tensor=sImageArray)
gModel = VGG16(include_top=False, weights='imagenet', input_tensor=rimgplaceholder)
cLayerName = 'block4_conv2'
sLayerNames = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                ]

yuckyi = 0
def callback(x):
    global yuckyi
    if yuckyi % 20 == 0:
        xOut = postprocess_array(x)
        xIm = save_original_size(xOut, index=yuckyi)
    yuckyi = yuckyi+1

P = get_feature_reps(x=cImageArray, layer_names=[cLayerName], model=cModel)[0]
As = get_feature_reps(x=sImageArray, layer_names=sLayerNames, model=sModel)
ws = np.ones(len(sLayerNames))/float(len(sLayerNames))

iterations = 600
x_val = rimg.flatten()


xopt, f_val, info= fmin_l_bfgs_b(calculate_loss, x_val, fprime=get_grad,
                            maxiter=iterations, disp=True, callback=callback)

xOut = postprocess_array(xopt)
xIm = save_original_size(xOut)

