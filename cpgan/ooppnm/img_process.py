import numpy as np
import porespy as ps
from skimage.measure import euler_number
from skimage.filters import threshold_multiotsu
from scipy.signal import medfilt
import torch


class Image_process:

    def __init__(self,PATH=None,im=None):
    
        self.im = im
        if PATH is not None:
            self.im = np.load(PATH)
    
    def phi(self,im_input):
        return ps.metrics.porosity(im_input)
    
    def eul(self,im,conn=3):
        # im: numpy array 
        conn = euler_number(im,connectivity=conn)
        return conn

    def spec_suf_area(self,im,domain_size=128):
        sf_a = ps.metrics.region_surface_areas(im)
        sf_a = sf_a/(domain_size**3)
        return sf_a[0]

    def two_point_corr(self,im,bins=20):
        # data = ps.metrics.two_point_correlation(im,bins=bins)
        data = ps.metrics.two_point_correlation_fft(im)
        return data

    # this function is to segment the image vertically
    def vertical_phi(self,segment_num=4):
        # input image dimension should be 3D cubic
            img_segment = []
            img_size = self.im.shape[0]
            segment_size = int( img_size/segment_num )

            start_index = 0
            end_index = 0+segment_size

            for i in range(segment_num):
                img_segment.append( self.im[:,:,start_index:end_index] )
                start_index += segment_size
                end_index += segment_size

            phi_component = []
            for i in img_segment:
                phi_component.append(self.phi(i))
            
            return phi_component

    
    def clean_img(self,tensor_img):
        '''
        1. detach tensor from GPU to cpu
        2  3^3 median image filter
        3 Otsu binary segmentation
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/median.htm
        https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html
        '''
        # detach tensor from GPU to cpu
        tensor_img = tensor_img.detach().cpu()
        f_images = tensor_img.numpy()

        # median filter
        img_filt = []
        for i in range(f_images.shape[0]):
            img_t = f_images[i][0]
            # img_t = medfilt(img_t,kernel_size=[3,3,3])
            thresholds = threshold_multiotsu(img_t,classes=2)
            img_t = np.digitize(img_t, bins=thresholds)
            img_filt.append(img_t)

        img_filt = np.array(img_filt)
        
        return img_filt
    

    def clean_img_filt(self,tensor_img):
        '''
        1. detach tensor from GPU to cpu
        2  3^3 median image filter
        3 Otsu binary segmentation
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/median.htm
        https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html
        '''
        # detach tensor from GPU to cpu
        tensor_img = tensor_img.detach().cpu()
        f_images = tensor_img.numpy()

        # median filter
        img_filt = []
        for i in range(f_images.shape[0]):
            img_t = f_images[i][0]
            img_t = medfilt(img_t,kernel_size=[3,3,3])
            thresholds = threshold_multiotsu(img_t,classes=2)
            img_t = np.digitize(img_t, bins=thresholds)
            img_filt.append(img_t)

        img_filt = np.array(img_filt)
        
        return img_filt



