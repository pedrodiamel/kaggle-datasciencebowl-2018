
from itertools import product
import skimage.morphology as morph
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
from scipy.stats import itemfreq
from skimage.color import label2rgb
import numpy as np
import cv2
import skfmm

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def watershed_center(image, center):
    distance = ndi.distance_transform_edt(image)
    markers, nr_blobs = ndi.label(center)
    labeled = morph.watershed(-distance, markers, mask=image)

    dropped, _ = ndi.label(image - (labeled > 0))
    dropped = np.where(dropped > 0, dropped + nr_blobs, 0)
    correct_labeled = dropped + labeled
    return relabel(correct_labeled)


def watershed_contour(image, contour):
    mask = np.where(contour == 1, 0, image)

    distance = ndi.distance_transform_edt(mask)
    markers, nr_blobs = ndi.label(mask)
    labeled = morph.watershed(-distance, markers, mask=image)

    dropped, _ = ndi.label(image - (labeled > 0))
    dropped = np.where(dropped > 0, dropped + nr_blobs, 0)
    correct_labeled = dropped + labeled
    return relabel(correct_labeled)


def postprocess(image, contour):
    
    cleaned_mask = clean_mask(image, contour)
    good_markers = get_markers(cleaned_mask, contour)
    good_distance = get_distance(cleaned_mask)

    labels = morph.watershed(-good_distance, good_markers, mask=cleaned_mask)

    labels = add_dropped_water_blobs(labels, cleaned_mask)

    m_thresh = threshold_otsu(image)
    initial_mask_binary = (image > m_thresh).astype(np.uint8)
    labels = drop_artifacts_per_label(labels, initial_mask_binary)

    labels = drop_small(labels, min_size=20)
    labels = fill_holes_per_blob(labels)

    return labels

def mpostprocess(score):
    
    predition = np.argmax(score[:,:,:3], axis=2).astype('uint8') 
    predition = predition == 1

    if predition.sum() < 20:
        return np.zeros_like( predition ) # not predition !!!

    score_prob = sigmoid(score)
    m = score_prob[:,:,1]
    #c = score_prob[:,:,2]
    
    m_thresh = threshold_otsu(m)
    #c_thresh = threshold_otsu(c)
    m_b = m > m_thresh
    #c_b = c > c_thresh

    predition = ndi.binary_fill_holes(predition)
    predition = morph.opening(predition,morph.disk(1))

    mk, _ = ndi.label(predition)

    distance = ndi.distance_transform_edt(predition)
    labels = morph.watershed(-distance,  mk, mask=m_b  )

    if labels.sum() < 20:
        return np.zeros_like( labels ) # not predition !!!

    labels = add_dropped_water_blobs(labels, m_b)

    initial_mask_binary = (m_b).astype(np.uint8)
    labels = drop_artifacts_per_label(labels, initial_mask_binary)
    labels = drop_small(labels, min_size=20)

    if labels.sum() < 20:
        return np.zeros_like( labels ) # not predition !!!


    labels = fill_holes_per_blob(labels)
    labels = decompose(labels)

    #ellipse aproximate
    #elp = fit_ellipse(labels)
    #labels_elp = create_ellipses_mask(score_prob.shape, elp)
    #labels_elp = create_label(labels_elp)
    #labels_elp = decompose(labels_elp)

    labels = clean_label( labels )
    labels = create_label(labels )
    labels = decompose( labels )

    return labels


def mpostprocessthresh(score, prob=0.5):
    
    score_prob = sigmoid(score)
    predition = score_prob[:,:,1] > prob 

    labels, _ = ndi.label(predition)
    labels = decompose(labels)
    labels = clean_label( labels )
    labels = create_label(labels )
    labels = decompose( labels )

    return labels

def mpostprocessmax(score):
    
    predition = np.argmax(score[:,:,:3], axis=2).astype('uint8') 
    predition = predition == 1
    labels, _ = ndi.label(predition)
    labels = decompose(labels)
    labels = clean_label( labels )
    labels = create_label(labels )
    labels = decompose( labels )

    return labels


def clean_label( masks ):
        
    cln_mask = []
    mean_area, radio = mean_blob_size( masks.max(axis=0)  )
    for mask in masks:        
        mask = (mask>128).astype(np.uint8)                
        try:
            _,contours,_ = cv2.findContours(mask, 1, 2) 
            if len(contours) == 0: continue

            contour = contours[0]            
            if len(contour) < 5:
                continue
            
            area = cv2.contourArea(contour)
            if area <= 10 or (mean_area !=1 and mean_area/area < 0.2):  # skip ellipses smaller then 5x5
                continue
            
            epsilon = 0.1*cv2.arcLength(contour,True)
            contour_aprox = cv2.approxPolyDP(contour,epsilon,True)   
            cv2.fillPoly(mask, contour_aprox, 1)             
            cln_mask.append(mask)
                        
        except ValueError as e:
            pass
    
    return np.array(cln_mask)


def create_label( labels ):
    #classe 0 back
    c,m = labels.shape[:2]
    mlabel = np.zeros_like(labels)
    for i in range(c):
        mlabel[i,:,:] = labels[i,:,:]*(i+1)
        
    mlabel = np.max(mlabel,axis=0)
    return relabel(mlabel)

def fit_ellipse( masks ):
    
    ellipses = []    
    pi_4 = np.pi * 4  
    
    for mask in masks:        
        
        #mask = pad_mask(mask, 5)
        mask = (mask>128).astype(np.uint8)                
        try:
            _,contours,_ = cv2.findContours(mask, 1, 2) 
            contour = contours[0]
            
            if len(contour) < 5:
                continue
            
            area = cv2.contourArea(contour)
            if area <= 25:  # skip ellipses smaller then 5x5
                continue
                
            arclen = cv2.arcLength(contour, True)
            circularity = (pi_4 * area) / (arclen * arclen)
            
            ellipse = cv2.fitEllipse(contour)  
            area_ellpse = (ellipse[1][0]/2.0)*(ellipse[1][1]/2.0)*np.pi          
            
            #print(area,area_ellpse,area/area_ellpse)
            if area/area_ellpse < 0.50:
                #print(area,area_ellpse, area/area_ellpse, ellipse[1], ellipse[0])
                continue             
                       
            ellipses.append( (ellipse, area, area_ellpse, circularity) )
                        
        except ValueError as e:
            pass
    
    return np.array(ellipses)


def create_ellipses_mask(masksize, ellipses ):    
    n = len(ellipses)
    masks = np.zeros( (n,masksize[0],masksize[1]), dtype=np.uint8 ) 
    for k in range(n):       
        try:         
            ellipse = ellipses[k][0]
            mask = masks[k,:,:]
            #mask = pad_mask(mask, 5)   
            #cv2.ellipse(elp,ellipse,1,2)
            poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
            cv2.fillPoly(mask, [poly], 1)            
            #mask = crop_mask(mask, 5)
            masks[k,:,:] = mask    
        except ValueError as e:
            pass       
    return masks


def drop_artifacts_per_label(labels, initial_mask):
    labels_cleaned = np.zeros_like(labels)
    for i in range(1, labels.max() + 1):
        component = np.where(labels == i, 1, 0)
        component_initial_mask = np.where(labels == i, initial_mask, 0)
        component = drop_artifacts(component, component_initial_mask)
        labels_cleaned = labels_cleaned + component * i
    return labels_cleaned


def clean_mask(m, c):    
    # threshold
    m_thresh = threshold_otsu(m)
    c_thresh = threshold_otsu(c)
    m_b = m > m_thresh
    c_b = c > c_thresh

    # combine contours and masks and fill the cells
    m_ = np.where(m_b | c_b, 1, 0)
    m_ = ndi.binary_fill_holes(m_)

    # close what wasn't closed before
    area, radius = mean_blob_size(m_b)
    struct_size = int(1.25 * radius)
    struct_el = morph.disk(struct_size)
    m_padded = pad_mask(m_, pad=struct_size)
    m_padded = morph.binary_closing(m_padded, selem=struct_el)
    m_ = crop_mask(m_padded, crop=struct_size)

    # open to cut the real cells from the artifacts
    area, radius = mean_blob_size(m_b)
    struct_size = int(0.75 * radius)
    struct_el = morph.disk(struct_size)
    m_ = np.where(c_b & (~m_b), 0, m_)
    m_padded = pad_mask(m_, pad=struct_size)
    m_padded = morph.binary_opening(m_padded, selem=struct_el)
    m_ = crop_mask(m_padded, crop=struct_size)

    # join the connected cells with what we had at the beginning
    m_ = np.where(m_b | m_, 1, 0)
    m_ = ndi.binary_fill_holes(m_)

    # drop all the cells that weren't present at least in 25% of area in the initial mask
    m_ = drop_artifacts(m_, m_b, min_coverage=0.25)

    return m_


def get_markers(m_b, c):
    # threshold
    c_thresh = threshold_otsu(c)
    c_b = c > c_thresh

    mk_ = np.where(c_b, 0, m_b)

    area, radius = mean_blob_size(m_b)
    struct_size = int(0.25 * radius)
    struct_el = morph.disk(struct_size)
    m_padded = pad_mask(mk_, pad=struct_size)
    m_padded = morph.erosion(m_padded, selem=struct_el)
    mk_ = crop_mask(m_padded, crop=struct_size)
    mk_, _ = ndi.label(mk_)
    return mk_


def get_distance(m_b):
    distance = ndi.distance_transform_edt(m_b)
    return distance


def add_dropped_water_blobs(water, mask_cleaned):
    water_mask = (water > 0).astype(np.uint8)
    dropped = mask_cleaned - water_mask
    dropped, _ = ndi.label(dropped)
    dropped = np.where(dropped, dropped + water.max(), 0)
    water = water + dropped
    return water


def fill_holes_per_blob(image):
    image_cleaned = np.zeros_like(image)
    for i in range(1, image.max() + 1):
        mask = np.where(image == i, 1, 0)
        mask = ndi.morphology.binary_fill_holes(mask)
        image_cleaned = image_cleaned + mask * i
    return image_cleaned


def drop_artifacts(mask_after, mask_pre, min_coverage=0.5):
    connected, nr_connected = ndi.label(mask_after)
    mask = np.zeros_like(mask_after)
    for i in range(1, nr_connected + 1):
        conn_blob = np.where(connected == i, 1, 0)
        initial_space = np.where(connected == i, mask_pre, 0)
        blob_size = np.sum(conn_blob)
        initial_blob_size = np.sum(initial_space)
        coverage = float(initial_blob_size) / float(blob_size)
        if coverage > min_coverage:
            mask = mask + conn_blob
        else:
            mask = mask + initial_space
    return mask


def mean_blob_size(mask):
    labels, labels_nr = ndi.label(mask)
    if labels_nr < 2:
        mean_area = 1
        mean_radius = 1
    else:
        mean_area = int(itemfreq(labels)[1:, 1].mean())
        mean_radius = int(np.round(np.sqrt(mean_area / np.pi)))
    return mean_area, mean_radius


def pad_mask(mask, pad):
    if pad <= 1:
        pad = 2
    h, w = mask.shape
    h_pad = h + 2 * pad
    w_pad = w + 2 * pad
    mask_padded = np.zeros((h_pad, w_pad))
    mask_padded[pad:pad + h, pad:pad + w] = mask
    mask_padded[pad - 1, :] = 1
    mask_padded[pad + h + 1, :] = 1
    mask_padded[:, pad - 1] = 1
    mask_padded[:, pad + w + 1] = 1

    return mask_padded


def crop_mask(mask, crop):
    if crop <= 1:
        crop = 2
    h, w = mask.shape
    mask_cropped = mask[crop:h - crop, crop:w - crop]
    return mask_cropped


def drop_small(img, min_size):
    img = morph.remove_small_objects(img, min_size=min_size)
    return relabel(img)

def tolabel(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled

def relabel(img):
    h, w = img.shape
    relabel_dict = {}
    for i, k in enumerate(np.unique(img)):
        if k == 0:
            relabel_dict[k] = 0
        else:
            relabel_dict[k] = i
    for i, j in product(range(h), range(w)):
        img[i, j] = relabel_dict[img[i, j]]
    return img

def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)
    if not masks: return np.array([labeled])
    else: return np.array(masks)


def mpostprocess_soft( softpred, line_width = 4 ):
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

