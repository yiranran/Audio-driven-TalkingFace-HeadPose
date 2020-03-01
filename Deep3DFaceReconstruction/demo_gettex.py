import tensorflow as tf
import os
from scipy.io import loadmat,savemat
from load_data import *
from reconstruct_mesh import Reconstruction_for_render, Render_layer, Render_layer2
from reconstruct_mesh import Reconstruction_for_render_new, Reconstruction_for_render_new_given, Reconstruction_for_render_new_given2, Project_layer
import sys
import glob
import pdb
import time
import cv2
import random
from preprocess_img import Preprocess

class RenderObject(object):
    def __init__(self, sess):
        if not os.path.isfile('./BFM/BFM_model_front.mat'):
            transferBFM09()
	    # read face model
        self.facemodel = BFM()

        self.faceshaper = tf.placeholder(name = "face_shape_r", shape = [1,35709,3], dtype = tf.float32)
        self.facenormr = tf.placeholder(name = "face_norm_r", shape = [1,35709,3], dtype = tf.float32)
        self.facecolor = tf.placeholder(name = "face_color", shape = [1,35709,3], dtype = tf.float32)
        self.rendered = Render_layer(self.faceshaper,self.facenormr,self.facecolor,self.facemodel,1)
        self.rendered2 = Render_layer2(self.faceshaper,self.facenormr,self.facecolor,self.facemodel,1)
        #self.project = Project_layer(self.faceshaper)

        self.rstimg = tf.placeholder(name = 'rstimg', dtype=tf.uint8)
        self.encode_png = tf.image.encode_png(self.rstimg)
        
        self.sess = sess
    
    def save_image(self, final_images, result_output_path):
        result_image = final_images[0, :, :, :]
        result_image = np.clip(result_image, 0., 1.).copy(order='C')
        #result_bytes = sess.run(tf.image.encode_png(result_image*255.0))
        result_bytes = self.sess.run(self.encode_png, {self.rstimg: result_image*255.0})
        with open(result_output_path, 'wb') as output_file:
            output_file.write(result_bytes)

    def save_image2(self, final_images, result_output_path, tx=0, ty=0):
        result_image = final_images[0, :, :, :]
        result_image = np.clip(result_image, 0., 1.) * 255.0
        result_image = np.round(result_image).astype(np.uint8)
        im = Image.fromarray(result_image,'RGBA')
        if tx != 0 or ty != 0:
            im = im.transform(im.size, Image.AFFINE, (1, 0, tx, 0, 1, ty))
        im.save(result_output_path)

    def show_clip_vertices(self, coef_path, clip_vertices, image_width=224, image_height=224):
        half_image_width = 0.5 * image_width
        half_image_height = 0.5 * image_height
        im = cv2.imread(coef_path.replace('coeff','render')[:-4]+'.png')
        for i in range(clip_vertices.shape[1]):
            if clip_vertices.shape[2] == 4:
                v0x = clip_vertices[0,i,0]
                v0y = clip_vertices[0,i,1]
                v0w = clip_vertices[0,i,3]
                px = int(round((v0x / v0w + 1.0) * half_image_width))
                py = int(image_height -1 - round((v0y / v0w + 1.0) * half_image_height))
            elif clip_vertices.shape[2] == 2:
                px = int(round(clip_vertices[0,i,0]))
                py = int(round(clip_vertices[0,i,1]))
            if px >= 0 and px < image_width and py >= 0 and py < image_height:
                cv2.circle(im, (px, py), 1, (0, 255, 0), -1)
        cv2.imwrite('show_clip_vertices.png',im)
    
    def gettexture(self, coef_path):
        data = loadmat(coef_path)
        coef = data['coeff']
        img_path = coef_path.replace('coeff','render')[:-4]+'.png'
        face_shape_r,face_norm_r,face_color,face_color2,face_texture2,tri,face_projection = Reconstruction_for_render_new(coef,self.facemodel,img_path)
        np.save(coef_path[:-4]+'_tex2.npy',face_texture2)
        return coef_path[:-4]+'_tex2.npy', face_texture2

    def gettexture2(self, coef_path):
        data = loadmat(coef_path)
        coef = data['coeff']
        img_path = coef_path[:-4]+'.jpg'
        face_shape_r,face_norm_r,face_color,face_color2,face_texture2,tri,face_projection = Reconstruction_for_render_new(coef,self.facemodel,img_path)
        np.save(coef_path[:-4]+'_tex2.npy',face_texture2)
        return coef_path[:-4]+'_tex2.npy'

    def render224_new(self, coef_path, result_output_path, tex2_path):
        if not os.path.exists(coef_path):
            return
        if os.path.exists(result_output_path):
            return
        data = loadmat(coef_path)
        coef = data['coeff']
        
        if not os.path.exists(os.path.dirname(result_output_path)):
            os.makedirs(os.path.dirname(result_output_path))
        
        face_shape_r,face_norm_r,face_color2,tri = Reconstruction_for_render_new_given(coef,self.facemodel,tex2_path)
        final_images = self.sess.run(self.rendered, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color2.astype('float32')})
        self.save_image(final_images, result_output_path)

    def render224_new2(self, coef_path, result_output_path, tex2):
        if not os.path.exists(coef_path):
            return
        #if os.path.exists(result_output_path):
        #    return
        data = loadmat(coef_path)
        coef = data['coeff']
        
        if not os.path.exists(os.path.dirname(result_output_path)):
            os.makedirs(os.path.dirname(result_output_path))
        
        face_shape_r,face_norm_r,face_color2,tri = Reconstruction_for_render_new_given2(coef,self.facemodel,tex2)
        final_images = self.sess.run(self.rendered, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color2.astype('float32')})
        self.save_image(final_images, result_output_path)
    
    def render224(self, coef_path, result_output_path):
        if not os.path.exists(coef_path):
            return
        if os.path.exists(result_output_path):
            return
        data = loadmat(coef_path)
        coef = data['coeff']
        
        if not os.path.exists(os.path.dirname(result_output_path)):
            os.makedirs(os.path.dirname(result_output_path))
        
        #t00 = time.time()
        face_shape_r,face_norm_r,face_color,tri = Reconstruction_for_render(coef,self.facemodel)
        final_images = self.sess.run(self.rendered, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color.astype('float32')})
        #t01 = time.time()
        self.save_image(final_images, result_output_path)
        #print(t01-t00,time.time()-t01)

    def render256(self, coef_path, savedir):
        data = loadmat(coef_path)
        coef = data['coeff']
        
        basen = os.path.basename(coef_path)[:-4]
        result_output_path = os.path.join(savedir,basen+'_render256.png')
        if not os.path.exists(os.path.dirname(result_output_path)):
            os.makedirs(os.path.dirname(result_output_path))
        
        face_shape_r,face_norm_r,face_color,tri = Reconstruction_for_render(coef,self.facemodel)
        final_images = self.sess.run(self.rendered2, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color.astype('float32')})
        self.save_image(final_images, result_output_path)

if __name__ == '__main__':
    with tf.Session() as sess:
        render_object = RenderObject(sess)
        coef_path = sys.argv[1]
        tex2_path,face_texture2 = render_object.gettexture(coef_path)
        result_output_path = coef_path.replace('output/coeff','output/render')[:-4]+'_rendernew.png'
        rp = render_object.render224_new2(coef_path,result_output_path,face_texture2)