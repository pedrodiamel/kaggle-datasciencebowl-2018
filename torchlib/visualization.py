
import cv2
import numpy as np
import matplotlib.pyplot as plt
import deep.datasets.imageutl as imutl


def torgb(im):
    if len(im.shape)==2:
        im = np.expand_dims(im, axis=2) 
        im = np.concatenate( (im,im,im), axis=2 )
    return im
        
def setcolor(im, mask, color):    
    tmp=im.copy()
    tmp=np.reshape( tmp, (-1, im.shape[2])  )   
    mask = np.reshape( mask, (-1,1))      
    tmp[ np.where(mask>0)[0] ,:] = color
    im=np.reshape( tmp, (im.shape)  )
    return im

def lincomb(im1,im2,mask, alpha):
    
    #im = np.zeros( (im1.shape[0], im1.shape[1], 3) )
    im = im1.copy()    
    
    row, col = np.where(mask>0)
    for i in range( len(row) ):
        r,c = row[i],col[i]
        #print(r,c)
        im[r,c,0] = im1[r,c,0]*(1-alpha) + im2[r,c,0]*(alpha)
        im[r,c,1] = im1[r,c,1]*(1-alpha) + im2[r,c,1]*(alpha)
        im[r,c,2] = im1[r,c,2]*(1-alpha) + im2[r,c,2]*(alpha)
    return im

def makebackgroundcell(labels):
    ch = labels.shape[2]
    cmap = plt.get_cmap('jet_r')
    imlabel = np.zeros( (labels.shape[0], labels.shape[1], 3) )    
    for i in range(ch):
        mask  = labels[:,:,i]
        color = cmap(float(i)/ch)
        imlabel = setcolor(imlabel,mask,color[:3])
    return imlabel

def makeedgecell(labels):
    ch = labels.shape[2]
    cmap = plt.get_cmap('jet_r')
    imedge = np.zeros( (labels.shape[0], labels.shape[1], 3) )    
    for i in range(ch):
        mask  = labels[:,:,i]
        color = cmap(float(i)/ch)
        mask = mask.astype(np.uint8)
        _,contours,_ = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE )
        for cnt in contours: cv2.drawContours(imedge, cnt, -1, color[:3], 1)
    return imedge

def makeimagecell(image, labels, alphaback=0.3, alphaedge=0.3):
    
    imagecell = image.copy()
    imagecell = imagecell - np.min(imagecell)
    imagecell = imagecell / np.max(imagecell)    
    imagecell = torgb(imagecell)
     
    mask  = np.sum(labels, axis=2)
    imagecellbackground = makebackgroundcell(labels)
    imagecelledge = makeedgecell(labels)
    maskedge = np.sum(imagecelledge, axis=2)
    
    imagecell = lincomb(imagecell,imagecellbackground, mask, alphaback )
    imagecell = lincomb(imagecell,imagecelledge, maskedge, alphaedge )
            
    return imagecell

def display_samplas(dataloader, row=3,col=3, alphaback=0.3, alphaedge=0.2):
    """
    Display random data from dataset
    For debug only
    """
    fig, ax = plt.subplots(row, col, figsize=(8, 8), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})
    
    for i in range(row):
        for j in range(col):
            
            k = np.random.randint( len(dataloader) )
            image, labels = dataloader[ k ]
            imagecell = makeimagecell(image, labels, alphaback=alphaback, alphaedge=alphaedge)
            
            ax[i,j].imshow(imagecell)
            ax[i,j].set_title('Image Idx: %d' % (k,))

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()