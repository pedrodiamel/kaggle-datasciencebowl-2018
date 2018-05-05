
import numpy as np
import scipy.misc
import skfmm
from sklearn.cluster import MiniBatchKMeans
import cv2


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def summary(data):
    print(np.min(data), np.max(data), data.shape)
    
def normalize(data):
    data = data - np.min(data)
    data = data / np.max(data)  
    return data

def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)

    if not masks: return [labeled]
    else: return masks


# def rle_encode(img):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     pixels = img.flatten()
#     pixels[0] = 0
#     pixels[-1] = 0
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
#     runs[1::2] = runs[1::2] - runs[:-1:2]
#     return ' '.join(str(x) for x in runs)


def rle_encode(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    if len(rle) != 0 and rle[-1] + rle[-2] == x.size:
        rle[-2] = rle[-2] - 1

    return rle

# def run_decode(rle, H, W, fill_value=255):
    
#     mask = np.zeros((H * W), np.uint8)
#     rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
#     for r in rle:
#         start = r[0]-1
#         end = start + r[1]
#         mask[start : end] = fill_value
#     mask = mask.reshape(W, H).T # H, W need to swap as transposing.
#     return mask


 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle #mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)



def process_data( softpred, line_width = 4 ):
    '''
    Precess data
    '''
    # assume the only output is a CHW image where C is the number
    # of classes, H and W are the height and width of the image

    # retain only the top class for each pixel
    class_data = np.argmax(softpred, axis=2).astype('uint8')

    # remember the classes we found
    found_classes = np.unique(class_data)

    fill_data = np.ndarray((class_data.shape[0], class_data.shape[1], 4), dtype='uint8')
    for x in range(3):
        fill_data[:, :, x] = class_data.copy()

    # Assuming that class 0 is the background
    mask = np.greater(class_data, 0)
    fill_data[:, :, 3] = mask * 255
    line_data = fill_data.copy()
    seg_data = fill_data.copy()
    
    # Black mask of non-segmented pixels
    mask_data = np.zeros(fill_data.shape, dtype='uint8')
    mask_data[:, :, 3] = (1 - mask) * 255

    # Generate outlines around segmented classes
    if len(found_classes) > 1:
        
        # Assuming that class 0 is the background.
        line_mask = np.zeros(class_data.shape, dtype=bool)
        max_distance = np.zeros(class_data.shape, dtype=float) + 1
        for c in (x for x in found_classes if x != 0):
            c_mask = np.equal(class_data, c)
            # Find the signed distance from the zero contour
            distance = skfmm.distance(c_mask.astype('float32') - 0.5)
            # Accumulate the mask for all classes
            line_mask |= c_mask & np.less(distance, line_width)
            max_distance = np.maximum(max_distance, distance + 128)

            line_data[:, :, 3] = line_mask * 255
            max_distance = np.maximum(max_distance, np.zeros(max_distance.shape, dtype=float))
            max_distance = np.minimum(max_distance, np.zeros(max_distance.shape, dtype=float) + 255)
            seg_data[:, :, 3] = max_distance

    return {
        'prediction':class_data,
        'line_data': line_data,
        'fill_data': fill_data,
        'seg_data' : seg_data,
    }



def quantized(imagein, k=5):
    
    h,w = imagein.shape[:2]
    image = cv2.cvtColor(imagein, cv2.COLOR_RGB2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters =  k )
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)

    return quant