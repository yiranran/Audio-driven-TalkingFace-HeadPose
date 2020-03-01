import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import models
import pdb

class Memory_Network(nn.Module):
    
    def __init__(self, mem_size, color_feat_dim = 512, spatial_feat_dim = 512, top_k = 256, alpha = 0.1, age_noise = 4.0, gpu_ids = []):
        
        super(Memory_Network, self).__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.ResNet18 = ResNet18().to(self.device)
        self.ResNet18 = self.ResNet18.eval()
        self.mem_size = mem_size
        self.color_feat_dim = color_feat_dim
        self.spatial_feat_dim = spatial_feat_dim
        self.alpha = alpha
        self.age_noise = age_noise
        self.top_k = top_k
        
        ## Each color_value is probability distribution
        self.color_value = F.normalize(random_uniform((self.mem_size, self.color_feat_dim), 0, 0.01), p = 1, dim=1).to(self.device)
        
        self.spatial_key = F.normalize(random_uniform((self.mem_size, self.spatial_feat_dim), -0.01, 0.01), dim=1).to(self.device)
        self.age = torch.zeros(self.mem_size).to(self.device)
        
        self.top_index = torch.zeros(self.mem_size).to(self.device)
        self.top_index = self.top_index - 1.0
        
        self.color_value.requires_grad = False
        self.spatial_key.requires_grad = False
        
        self.Linear = nn.Linear(512, spatial_feat_dim)
        self.body = [self.ResNet18, self.Linear]
        self.body = nn.Sequential(*self.body)
        self.body = self.body.to(self.device)
        
    def forward(self, x):
        q = self.body(x)
        q = F.normalize(q, dim = 1)
        return q
    
    def unsupervised_loss(self, query, color_feat, color_thres):
        
        bs = query.size()[0]
        cosine_score = torch.matmul(query, torch.t(self.spatial_key))
        
        top_k_score, top_k_index = torch.topk(cosine_score, self.top_k, 1)
        
        ### For unsupervised training
        color_value_expand = torch.unsqueeze(torch.t(self.color_value), 0)
        color_value_expand = torch.cat([color_value_expand[:,:,idx] for idx in top_k_index], dim = 0)
        
        color_feat_expand = torch.unsqueeze(color_feat, 2)
        color_feat_expand = torch.cat([color_feat_expand for _ in range(self.top_k)], dim = 2)
        
        #color_similarity = self.KL_divergence(color_value_expand, color_feat_expand, 1)
        color_similarity = torch.sum(torch.mul(color_value_expand, color_feat_expand),dim=1)
        
        #loss_mask = color_similarity < color_thres
        loss_mask = color_similarity > color_thres
        loss_mask = loss_mask.float()
        
        pos_score, pos_index = torch.topk(torch.mul(top_k_score, loss_mask), 1, dim = 1)
        neg_score, neg_index = torch.topk(torch.mul(top_k_score, 1 - loss_mask), 1, dim = 1)
        
        loss = self._unsupervised_loss(pos_score, neg_score)
        
        return loss
    
    
    def memory_update(self, query, color_feat, color_thres, top_index):
        
        cosine_score = torch.matmul(query, torch.t(self.spatial_key))
        top1_score, top1_index = torch.topk(cosine_score, 1, dim = 1)
        top1_index = top1_index[:, 0]
        top1_feature = self.spatial_key[top1_index]
        top1_color_value = self.color_value[top1_index]
        
        #color_similarity1 = self.KL_divergence(top1_color_value, color_feat, 1)
        color_similarity = torch.sum(torch.mul(top1_color_value, color_feat),dim=1)
            
        #memory_mask = color_similarity < color_thres
        memory_mask = color_similarity > color_thres
        self.age = self.age + 1.0
        
        ## Case 1 update
        case_index = top1_index[memory_mask]
        self.spatial_key[case_index] = F.normalize(self.spatial_key[case_index] + query[memory_mask], dim = 1)
        self.age[case_index] = 0.0
        #if torch.sum(memory_mask).cpu().numpy()==1:
        #    print(top_index,'update',self.top_index[case_index],color_similarity)
        
        ## Case 2 replace
        memory_mask = 1.0 - memory_mask
        case_index = top1_index[memory_mask]
        
        random_noise = random_uniform((self.mem_size, 1), -self.age_noise, self.age_noise)[:, 0]
        random_noise = random_noise.to(self.device)
        age_with_noise = self.age + random_noise
        old_values, old_index = torch.topk(age_with_noise, len(case_index), dim=0)
        
        self.spatial_key[old_index] = query[memory_mask]
        self.color_value[old_index] = color_feat[memory_mask]
        #if torch.sum(memory_mask).cpu().numpy()==1:
        #    print(top_index[memory_mask],'replace',self.top_index[old_index],color_similarity)
        #pdb.set_trace()
        self.top_index[old_index] = top_index[memory_mask]
        self.age[old_index] = 0.0
        
        return torch.sum(memory_mask).cpu().numpy()==1 # for batch size 1, return number of replace
    
    def topk_feature(self, query, top_k = 1):
        _bs = query.size()[0]
        cosine_score = torch.matmul(query, torch.t(self.spatial_key))
        topk_score, topk_index = torch.topk(cosine_score, top_k, dim = 1)
        
        topk_feat = torch.cat([torch.unsqueeze(self.color_value[topk_index[i], :], dim = 0) for i in range(_bs)], dim = 0)
        topk_idx = torch.cat([torch.unsqueeze(self.top_index[topk_index[i]], dim = 0) for i in range(_bs)], dim = 0)
        
        return topk_feat, topk_idx, topk_index

    def get_feature(self, k, _bs):
        feat = torch.cat([torch.unsqueeze(self.color_value[k, :], dim = 0) for i in range(_bs)], dim = 0)
        return feat, self.top_index[k]

    
    def KL_divergence(self, a, b, dim, eps = 1e-8):
        
        b = b + eps
        log_val = torch.log10(torch.div(a, b))
        kl_div = torch.mul(a, log_val)
        kl_div = torch.sum(kl_div, dim = dim)
        
        return kl_div
        
        
    def _unsupervised_loss(self, pos_score, neg_score):
        
        hinge = torch.clamp(neg_score - pos_score + self.alpha, min = 0.0)
        loss = torch.mean(hinge)
        
        return loss
        

def random_uniform(shape, low, high):
    x = torch.rand(*shape)
    result = (high - low) * x + low
    
    return result
    

class ResNet18(nn.Module):
    def __init__(self, pre_trained = True, require_grad = False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained = True)       
        
        self.body = [layers for layers in self.model.children()]
        self.body.pop(-1)
        
        self.body = nn.Sequential(*self.body)
        
        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False
                
    def forward(self, x):
        x = self.body(x)
        x = x.view(-1, 512)
        return x