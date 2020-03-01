import torch
import torch.nn as nn
# from pts3d import *
from ops import *
import torchvision.models as models
import functools
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from convolutional_rnn import Conv2dGRU

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class AT_net(nn.Module):
    def __init__(self):
        super(AT_net, self).__init__()
        self.lmark_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),

            )
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
       
            )
        self.lstm = nn.LSTM(256*3,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,6),
            )

    def forward(self, example_landmark, audio):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        example_landmark_f = self.lmark_encoder(example_landmark)
        #print 'example_landmark_f', example_landmark_f.shape # (1,512)
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature)
            features = torch.cat([example_landmark_f,  current_feature], 1)
            #print 'current_feature', current_feature.shape # (1,256)
            #print 'features', features.shape # (1,768)
            lstm_input.append(features)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input, hidden)
        fc_out   = []
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]
            fc_out.append(self.lstm_fc(fc_in))
        return torch.stack(fc_out, dim = 1)

class ATC_net(nn.Module):
    def __init__(self, para_dim):
        super(ATC_net, self).__init__()
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
       
            )
        self.lstm = nn.LSTM(256,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,para_dim),
            )

    def forward(self, audio):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature)
            lstm_input.append(current_feature)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input, hidden) # output, (hn,cn) = LSTM(input, (h0,c0))
        fc_out   = []
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]
            fc_out.append(self.lstm_fc(fc_in))
        return torch.stack(fc_out, dim = 1)


class AT_single(nn.Module):
    def __init__(self):
        super(AT_single, self).__init__()
        # self.lmark_encoder = nn.Sequential(
        #     nn.Linear(6,256),
        #     nn.ReLU(True),
        #     nn.Linear(256,512),
        #     nn.ReLU(True),

        #     )
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1,normalizer = None),
            conv2d(64,128,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1,normalizer = None),
            conv2d(256,256,3,1,1,normalizer = None),
            conv2d(256,512,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
            nn.Linear(256, 6)
            )
        # self.fusion = nn.Sequential(
        #     nn.Linear(256 *3, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 6)
        #     )

    def forward(self, audio):
        current_audio = audio.unsqueeze(1)
        current_feature = self.audio_eocder(current_audio)
        current_feature = current_feature.view(current_feature.size(0), -1)

        output = self.audio_eocder_fc(current_feature)
     
        return output


class GL_Discriminator(nn.Module):


    def __init__(self):
        super(GL_Discriminator, self).__init__()

        self.image_encoder_dis = nn.Sequential(
            conv2d(3,64,3,2, 1,normalizer=None),
            # conv2d(64, 64, 4, 2, 1),
            conv2d(64, 128, 3, 2, 1),

            conv2d(128, 256, 3, 2, 1),

            conv2d(256, 512, 3, 2, 1),
            )
        self.encoder = nn.Sequential(
            nn.Linear(136, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 136),
            nn.Tanh()
            )
        self.img_fc = nn.Sequential(
            nn.Linear(512*8*8, 512),
            nn.ReLU(True),
            )

        self.lstm = nn.LSTM(1024,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,136),
            nn.Tanh())
        self.decision = nn.Sequential(
            nn.Linear(256,1),
            )
        self.aggregator = nn.AvgPool1d(kernel_size = 16)
        self.activate = nn.Sigmoid()
    def forward(self, xs, example_landmark):
        hidden = ( torch.autograd.Variable(torch.zeros(3, example_landmark.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, example_landmark.size(0), 256).cuda()))
        lstm_input = list()
        lmark_feature= self.encoder(example_landmark)
        for step_t in range(xs.size(1)):
            x = xs[:,step_t,:,:, :]
            x.data = x.data.contiguous()
            x = self.image_encoder_dis(x)
            x = x.view(x.size(0), -1)
            x = self.img_fc(x)
            new_feature = torch.cat([lmark_feature, x], 1)
            lstm_input.append(new_feature)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input, hidden)
        fc_out   = []
        decision = []
        for step_t in range(xs.size(1)):
            fc_in = lstm_out[:,step_t,:]
            decision.append(self.decision(fc_in))
            fc_out.append(self.lstm_fc(fc_in)+ example_landmark)
        fc_out = torch.stack(fc_out, dim = 1)
        decision = torch.stack(decision, dim = 2)
        decision = self.aggregator(decision)
        decision = self.activate(decision)
        return decision.view(decision.size(0)), fc_out



class VG_net(nn.Module):
    def __init__(self,input_nc = 3, output_nc = 3,ngf = 64, use_dropout=True, use_bias=False,norm_layer=nn.BatchNorm2d,n_blocks = 9,padding_type='zero'):
        super(VG_net,self).__init__()
        dtype            = torch.FloatTensor


        self.image_encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            conv2d(3, 64, 7,1, 0),

            # conv2d(64,16,3,1,1),
            conv2d(64,64,3,2,1),
            # conv2d(32,64,3,1,1),
            conv2d(64,128,3,2,1)
            )

        self.image_encoder2 = nn.Sequential(
            conv2d(128,256,3,2,1),
            conv2d(256,512,3,2,1)
            )

        self.landmark_encoder  = nn.Sequential(
            nn.Linear(136, 64),
            nn.ReLU(True)
            )

        self.landmark_encoder_stage2 = nn.Sequential(
            conv2d(1,256,3),
            
            )
        self.lmark_att = nn.Sequential(
            nn.ConvTranspose2d(512, 256,kernel_size=3, stride=(2),padding=(1), output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128,kernel_size=3, stride=(2),padding=(1), output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            conv2d(128, 1,3, activation=nn.Sigmoid, normalizer=None)
            )
        self.lmark_feature = nn.Sequential(
            conv2d(256,512,3)) 
     
        model = []
        n_downsampling = 4
        mult = 2**(n_downsampling -1  )
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling ):
            mult = 2**(n_downsampling-i-1 ) 
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            if i == n_downsampling-3:
                self.generator1 = nn.Sequential(*model)
                model = []

        self.base = nn.Sequential(*model)
        model = []
        model += [nn.Conv2d(int(ngf/2), output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]
        self.generator_color = nn.Sequential(*model)

        model = []
        model += [nn.Conv2d(int(ngf/2), 1, kernel_size=7, padding=3)]
        model += [nn.Sigmoid()]
        self.generator_attention = nn.Sequential(*model)

        self.bottle_neck = nn.Sequential(conv2d(1024,128,3,1,1))

        
        self.convGRU = Conv2dGRU(in_channels = 128, out_channels = 512, kernel_size = (3), num_layers = 1, bidirectional = False, dilation = 2, stride = 1, dropout = 0.5 )
        
    def forward(self,image, landmarks, example_landmark ):
        # ex_landmark1 = self.landmark_encoder(example_landmark.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128,128))
        image_feature1 = self.image_encoder1(image)
        image_feature = self.image_encoder2(image_feature1)
        ex_landmark1 = self.landmark_encoder(example_landmark.view(example_landmark.size(0), -1))
        ex_landmark1 = ex_landmark1.view(ex_landmark1.size(0), 1, image_feature.size(2), image_feature.size(3) )
        ex_landmark1 = self.landmark_encoder_stage2(ex_landmark1)
        ex_landmark = self.lmark_feature(ex_landmark1)
        
        lstm_input = list()
        lmark_atts = list()
        for step_t in range(landmarks.size(1)):
            landmark = landmarks[:,step_t,:]
            landmark.data = landmark.data.contiguous()
            landmark = self.landmark_encoder(landmark.view(landmark.size(0), -1))
            landmark = landmark.view(landmark.size(0), 1, image_feature.size(2), image_feature.size(3) )
            landmark = self.landmark_encoder_stage2(landmark)

            lmark_att = self.lmark_att( torch.cat([landmark, ex_landmark1], dim=1))
            landmark = self.lmark_feature(landmark)

            inputs =  self.bottle_neck(torch.cat([image_feature, landmark - ex_landmark], dim=1))
            lstm_input.append(inputs)
            lmark_atts.append(lmark_att)
        lmark_atts =torch.stack(lmark_atts, dim = 1)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_output, _ = self.convGRU(lstm_input)

        outputs = []
        atts = []
        colors = []
        for step_t in range(landmarks.size(1)):
            input_t = lstm_output[:,step_t,:,:,:]
            v_feature1 = self.generator1(input_t)
            v_feature1_f = image_feature1 * (1- lmark_atts[:,step_t,:,:,:] ) + v_feature1 * lmark_atts[:,step_t,:,:,:] 
            base = self.base(v_feature1_f)
            color = self.generator_color(base)
            att = self.generator_attention(base)
            atts.append(att)
            colors.append(color)
            output = att * color + (1 - att ) * image
            outputs.append(output)
        return torch.stack(outputs, dim = 1), torch.stack(atts, dim = 1), torch.stack(colors, dim = 1), lmark_atts


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out