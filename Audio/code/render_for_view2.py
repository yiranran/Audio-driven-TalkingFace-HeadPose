import tensorflow as tf
import os
from scipy.io import loadmat,savemat
from reconstruct_mesh import Reconstruction_for_render, Render_layer, Render_layer2
from reconstruct_mesh import Reconstruction_for_render_new_given
import numpy as np
import sys
import glob
from PIL import Image
import pdb

rootdir = '../../Deep3DFaceReconstruction/'

class BFM():
    def __init__(self):
        model_path = rootdir+'BFM/BFM_model_front.mat'
        model = loadmat(model_path)
        self.meanshape = model['meanshape'] # mean face shape 
        self.idBase = model['idBase'] # identity basis
        self.exBase = model['exBase'] # expression basis
        self.meantex = model['meantex'] # mean face texture
        self.texBase = model['texBase'] # texture basis
        self.point_buf = model['point_buf'] # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
        self.tri = model['tri'] # vertex index for each triangle face, starts from 1
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int32) - 1 # 68 face landmark index, starts from 0

class RenderObject(object):
    def __init__(self, sess):
	    # read face model
        self.facemodel = BFM()

        self.faceshaper = tf.placeholder(name = "face_shape_r", shape = [1,35709,3], dtype = tf.float32)
        self.facenormr = tf.placeholder(name = "face_norm_r", shape = [1,35709,3], dtype = tf.float32)
        self.facecolor = tf.placeholder(name = "face_color", shape = [1,35709,3], dtype = tf.float32)
        self.rendered = Render_layer(self.faceshaper,self.facenormr,self.facecolor,self.facemodel,1)
        self.rendered2 = Render_layer2(self.faceshaper,self.facenormr,self.facecolor,self.facemodel,1)

        self.rstimg = tf.placeholder(name = 'rstimg', dtype=tf.uint8)
        self.encode_png = tf.image.encode_png(self.rstimg)
        
        self.sess = sess
        self.last = np.zeros((6))
    
    def save_image(self, final_images, result_output_path):
        result_image = final_images[0, :, :, :]
        result_image = np.clip(result_image, 0., 1.).copy(order='C')
        result_bytes = sess.run(self.encode_png, {self.rstimg: result_image*255.0})
        with open(result_output_path, 'wb') as output_file:
            output_file.write(result_bytes)

    def save_image2(self, final_images, result_output_path, tx=0, ty=0):
        result_image = final_images[0, :, :, :]
        #print(result_image.shape)
        result_image = np.clip(result_image, 0., 1.) * 255.0
        result_image = np.round(result_image).astype(np.uint8)
        im = Image.fromarray(result_image,'RGBA')
        #pdb.set_trace()
        if tx != 0 or ty != 0:
            im = im.transform(im.size, Image.AFFINE, (1, 0, -tx, 0, 1, -ty))
        im.save(result_output_path)


    def render(self, coef_path):
        data = loadmat(coef_path)
        coef = data['coeff']
        
        result_output_path = os.path.join('output/render_fuse',coef_path[13:-4]+'_render.png')
        if not os.path.exists(os.path.dirname(result_output_path)):
            os.makedirs(os.path.dirname(result_output_path))
        
        face_shape_r,face_norm_r,face_color,tri = Reconstruction_for_render(coef,self.facemodel)
        final_images = self.sess.run(self.rendered, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color.astype('float32')})
        self.save_image(final_images, result_output_path)

    def render256(self, coef_path):
        data = loadmat(coef_path)
        coef = data['coeff']
        
        result_output_path = os.path.join('output/render_fuse',coef_path[13:-4]+'_render256.png')
        if not os.path.exists(os.path.dirname(result_output_path)):
            os.makedirs(os.path.dirname(result_output_path))
        
        face_shape_r,face_norm_r,face_color,tri = Reconstruction_for_render(coef,self.facemodel)
        final_images = self.sess.run(self.rendered2, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color.astype('float32')})
        self.save_image(final_images, result_output_path)
        

    def render2(self, coef_path1, coef_path2, result_output_path, pose=0, relativeframe=0, frame0=0, tran=0):
        data1 = loadmat(coef_path1)
        coef1 = data1['coeff']
        if coef_path2[-4:] == '.mat':
            data2 = loadmat(coef_path2)
            coef2 = data2['coeff']
            #transfer ex_coef
            coef1[:,80:144] = coef2[:,80:144]
        else:
            coef2 = np.load(coef_path2) # shape (64, )
            if pose == 0:
                coef1[:,80:144] = coef2
            else:
                L = 64
                coef1[:,80:144] = coef2[:L]
                if relativeframe == 0:#################
                    coef1[:,224:227] = coef2[L:L+3]
                    coef1[:,254:257] = coef2[L+3:L+6]
                    coef1[:,256] += 0.5
                elif relativeframe == 2:
                    coef1[:,224:227] = coef2[L:L+3]
                    coef1[:,254:257] = coef2[L+3:L+6]
                else:
                    if not frame0:
                        coef1[:,224:227] = coef2[L:L+3] + self.last[:3]
                        coef1[:,254:257] = coef2[L+3:L+6] + self.last[3:6]
                    self.last[:3] = coef1[:,224:227]
                    self.last[3:6] = coef1[:,254:257]

        face_shape_r,face_norm_r,face_color,tri = Reconstruction_for_render(coef1,self.facemodel)
        final_images = self.sess.run(self.rendered, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color.astype('float32')})
        if coef2.shape[0] >= L+8 and tran==1:
            self.save_image2(final_images, result_output_path, tx=coef2[L+6], ty=coef2[L+7]) 
        else:
            self.save_image(final_images, result_output_path)
    
    def render2_newtex(self, coef_path1, coef_path2, tex2_path, result_output_path, pose=0, relativeframe=0, frame0=0, tran=0):
        data1 = loadmat(coef_path1)
        coef1 = data1['coeff']
        if coef_path2[-4:] == '.mat':
            data2 = loadmat(coef_path2)
            coef2 = data2['coeff']
            #transfer ex_coef
            coef1[:,80:144] = coef2[:,80:144]
        else:
            coef2 = np.load(coef_path2) # shape (64, )
            L = 64
            coef1[:,80:144] = coef2[:L]
            if relativeframe == 2:
                coef1[:,224:227] = coef2[L:L+3]
                coef1[:,254:257] = coef2[L+3:L+6]
            if relativeframe == 0:
                coef1[:,224:227] = coef2[L:L+3]
                coef1[:,254:257] = coef2[L+3:L+6]
                coef1[:,256] += 0.5
        face_shape_r,face_norm_r,face_color2,tri = Reconstruction_for_render_new_given(coef1,self.facemodel,tex2_path)
        final_images = self.sess.run(self.rendered, feed_dict={self.faceshaper: face_shape_r.astype('float32'), self.facenormr: face_norm_r.astype('float32'), self.facecolor: face_color2.astype('float32')})
        if coef2.shape[0] >= L+8 and tran==1:
            self.save_image2(final_images, result_output_path, tx=coef2[L+6], ty=coef2[L+7]) 
        else:
            self.save_image(final_images, result_output_path)

if __name__ == '__main__':
    coef_dir = sys.argv[1]
    coef_path1 = sys.argv[2]
    save_dir = sys.argv[3]
    pose = int('pose' in coef_dir)
    relativeframe = int(sys.argv[4])
    tran = int(sys.argv[5])
    tex2_path = sys.argv[6] if len(sys.argv) > 6 else ''
    print('pose',pose)
    print('relativeframe',relativeframe)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    coef_paths = sorted(glob.glob(coef_dir+'/*.npy'))
    L = len(coef_paths)
    with tf.Session() as sess:
        render_object = RenderObject(sess)
        for i in range(L):
            basen = os.path.basename(coef_paths[i])
            save = os.path.join(save_dir,basen[:-4]+'.png')
            if tex2_path == '':
                # old texture
                #render_object.render2(coef_path1, coef_paths[i], save)
                render_object.render2(coef_path1, coef_paths[i], save, pose, relativeframe, i==0, tran=tran)
            else:
                # new texture
                render_object.render2_newtex(coef_path1, coef_paths[i], tex2_path, save, pose, relativeframe, i==0, tran=tran)
            if i % 100 == 0 and i != 0:
                print('rendered', i)