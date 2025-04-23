import torch
# from utils.expert import weight_sum_var, ivw_aggregate_var
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 

def compute_cosine_similarity(z_1, z_2):
    """
    计算共享特征和私有特征之间的正交损失
    使用线性变换：loss = 1 - (cos_similarity + 1)/2
    Args:
        z_sample: 共享特征 [batch_size, feature_dim]
        z_sample_list_p: 私有特征列表 list of [batch_size, feature_dim]
    Returns:
        orthogonal_loss: 正交损失
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    total_loss = 0
    cos_similarity = cos(z_1, z_2)  # [batch_size]
    # 使用线性变换将余弦相似度转换为损失
    loss = 1 - ((cos_similarity + 1) / 2)  # 映射到[0,1]区间
    total_loss += torch.mean(loss)
    return total_loss

def compute_cosine_similarity_list(z_sample_view_s, z_sample_view_p):
    """
    计算每个视图的共享特征和私有特征之间的余弦相似度，但排除z_sample_view_s的最后一个元素
    Args:
        z_sample_view_s: 视图共享特征 list of [batch_size, feature_dim]
        z_sample_view_p: 视图私有特征 list of [batch_size, feature_dim]
    Returns:
        cos_loss: 余弦相似度损失
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    total_loss = 0
    
    # 获取除最后一个元素外的z_sample_view_s
    z_sample_view_s_filtered = z_sample_view_s[:-1]
    
    # 确保过滤后的z_sample_view_s和z_sample_view_p长度相同
    assert len(z_sample_view_s_filtered) == len(z_sample_view_p), "过滤后的共享特征列表长度必须与私有特征列表长度相同"
    
    # 如果过滤后列表为空，返回0损失
    if len(z_sample_view_s_filtered) == 0:
        return torch.tensor(0.0).to(z_sample_view_s[0].device)
    
    # 对每个视图分别计算其共享特征和私有特征的余弦相似度
    for z_s, z_p in zip(z_sample_view_s_filtered, z_sample_view_p):
        cos_similarity = cos(z_s, z_p)  # [batch_size]
        # 使用线性变换将余弦相似度转换为损失
        loss = 1 - ((cos_similarity + 1) / 2)  # 映射到[0,1]区间
        total_loss += torch.mean(loss)
    
    return total_loss / len(z_sample_view_s_filtered)

def manual_gaussian_log_prob_stable(x, mu, var, eps=1e-8):
    """更稳定的手动高斯分布对数概率计算
    
    Args:
        x: 样本点 [batch_size, dim]
        mu: 均值 [batch_size, dim]
        var: 方差 [batch_size, dim]
        eps: 数值稳定性常数
    Returns:
        log_prob: 对数概率 [batch_size]
    """
    
    # 确保方差为正值
    var = torch.clamp(var, min=eps)
    
    # 计算常数项 -0.5 * log(2π)
    const = -0.5 * np.log(2 * np.pi)
    
    # 计算 -0.5 * log(σ²)，添加数值稳定性
    log_det = -0.5 * torch.log(var + eps)
    
    # 计算 -0.5 * (x - μ)²/σ²，使用更稳定的计算方式
    diff = x - mu
    # 限制diff的大小，防止极端值
    diff = torch.clamp(diff, min=-1e6, max=1e6)
    # 避免除以接近0的数
    scaled_diff = diff / torch.sqrt(var + eps)
    mahalanobis = -0.5 * scaled_diff.pow(2)
    
    # 每个维度的对数概率
    log_prob_per_dim = const + log_det + mahalanobis
    
    # 检查是否有无穷值或NaN
    # if torch.isnan(log_prob_per_dim).any() or torch.isinf(log_prob_per_dim).any():
    #     print("警告: 对数概率计算中出现NaN或Inf")
    #     log_prob_per_dim = torch.nan_to_num(log_prob_per_dim, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 在求和前检查数值范围，避免极端值
    log_prob_per_dim = torch.clamp(log_prob_per_dim, min=-100, max=100)
    
    # 独立多维高斯分布的对数概率是各维度对数概率之和
    log_prob = torch.sum(log_prob_per_dim, dim=-1) / 512
    
    # 最终检查
    if torch.isnan(log_prob).any():
        print("警告: 最终结果中包含NaN")
        log_prob = torch.zeros_like(log_prob)
    
    return log_prob


def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var)
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def fill_with_label(label_embedding,label,x_embedding,inc_V_ind):
    fea = label.matmul(label_embedding)/(label.sum(dim=1,keepdim=True)+1e-8)
    new_x =  x_embedding*inc_V_ind.T.unsqueeze(-1) + fea.unsqueeze(0)*(1-inc_V_ind.T.unsqueeze(-1))
    return new_x
class MLP(nn.Module):
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,dropout_rate=0.,final_act=True,final_norm=True):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm
        # init layers
        self.mlps =[]
        layers = []
        if len(hidden_dim)>0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            # layers.append(nn.Dropout(dropout_rate))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
            ##hidden layer
            for i in range(len(hidden_dim)-1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                # layers.append(nn.Dropout(dropout_rate))
                layers.append(self.norm(hidden_dim[i+1]))
                layers.append(self.act())
                self.mlps.append(nn.Sequential(*layers))
                layers = []
            ##output layer
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            # layers.append(nn.Dropout(dropout_rate))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
        self.mlps = nn.ModuleList(self.mlps)
    def forward(self, x):
        for layers in self.mlps:
            x = layers(x)
            # x = x + y
        return x
    
class sharedQz_inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(sharedQz_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())
        # self.qzv_inference = nn.Sequential(*self.qz_layer)
    def forward(self, x):
        hidden_features = self.mlp(x)
        z_mu = self.z_loc(hidden_features)
        z_sca = self.z_sca(hidden_features)
        # class_feature  = self.z
        return z_mu, z_sca
    
class inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        # self.qzv_inference = nn.Sequential(*self.qz_layer)
    def forward(self, x):
        hidden_features = self.mlp(x)
        # class_feature  = self.z
        return hidden_features
    
class Px_generation_mlp(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[512]):
        super(Px_generation_mlp, self).__init__()
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim,final_act=False,final_norm=False)
        # self.transfer_act = nn.ReLU
        # self.px_layer = mlp_layers_creation(self.z_dim, self.x_dim, self.layers, self.transfer_act)
        # self.px_z = nn.Sequential(*self.px_layer)
    def forward(self, z):
        xr = self.mlp(z)
        return xr

class VAE(nn.Module):
    def __init__(self, d_list,z_dim,class_num):
        super(VAE, self).__init__()
        self.x_dim_list = d_list
        self.k = class_num
        self.z_dim = z_dim
        self.num_views = len(d_list)

        self.mlp_2view = MLP(z_dim*4, z_dim)

    # self.switch_layers = switch_layers(z_dim,self.num_views)
        self.z_inference_s = []
        self.z_inference_p = []
        self.mapinference = []
        self.mu2 = nn.Parameter(torch.full((self.z_dim,),1.), requires_grad=False)
        self.sigma = nn.Parameter(torch.full((self.z_dim,),2.), requires_grad=False)
        self.prior2 = torch.distributions.Normal(loc=self.mu2, scale=self.sigma)
        self.prior2 = torch.distributions.Independent(self.prior2, 1)
        
        # 为每一个视图都创建一个独立的编码器
        for v in range(self.num_views):
            self.z_inference_s.append(inference_mlp(self.x_dim_list[v], self.z_dim))
            self.z_inference_p.append(inference_mlp(self.x_dim_list[v], self.z_dim))
            self.mapinference.append(inference_mlp(self.x_dim_list[v], self.z_dim))
        self.qz_inference_s = nn.ModuleList(self.z_inference_s)
        self.qz_inference_p = nn.ModuleList(self.z_inference_p)
        self.mapinference = nn.ModuleList(self.mapinference)
        self.qz_inference = nn.ModuleList([inference_mlp(self.x_dim_list[v], self.z_dim) for v in range(self.num_views)])
        # 定义共享潜在分布的推理模块,通过 MLP 提取特征，再通过线性层分别输出潜在分布的均值z_loc和方差z_sca(使用 Softplus 确保正性)
        self.qz_inference_header = sharedQz_inference_mlp(self.z_dim, self.z_dim)
        self.x_generation_s = []
        self.x_generation_p = []
        for v in range(self.num_views):
            # 为每一个视图都创建一个独立的解码器
            self.x_generation_s.append(Px_generation_mlp(self.z_dim,self.x_dim_list[v]))
            self.x_generation_p.append(Px_generation_mlp(self.z_dim,self.x_dim_list[v]))
        self.px_generation_s = nn.ModuleList(self.x_generation_s)
        self.px_generation_p = nn.ModuleList(self.x_generation_p)
        self.px_generation = nn.ModuleList([Px_generation_mlp(self.z_dim, self.x_dim_list[v]) for v in range(self.num_views)])
    def inference_z1(self, x_list):
        uniview_mu_s_list = []
        uniview_sca_s_list = []
        fea_s_list = []
        fea_p_list = []
        for v in range(self.num_views):
            if torch.sum(torch.isnan(x_list[v])).item() > 0:
                print("zzz:nan")
                pass
            # 每一个view都通过qz_inference提取特征
            fea_s = self.qz_inference_s[v](x_list[v])
            fea_p = self.qz_inference_p[v](x_list[v])   
            fea_s_list.append(fea_s)
            fea_p_list.append(fea_p)
        fea_concat = torch.cat(fea_s_list[:-1], dim=1)
        mapped_fea = self.mlp_2view(fea_concat)
        fea_s_list[-1] = mapped_fea
        for fea_s in (fea_s_list):
            z_mu_v_s, z_sca_v_s = self.qz_inference_header(fea_s)
            uniview_mu_s_list.append(z_mu_v_s)
            uniview_sca_s_list.append(z_sca_v_s)
        return uniview_mu_s_list, uniview_sca_s_list, fea_p_list
    
    def inference_z2(self, x_list, map_fea):
        uniview_mu_s_list = []
        uniview_sca_s_list = []
        fea_s_list = []
        fea_p_list = []
        for v in range(self.num_views):
            if torch.sum(torch.isnan(x_list[v])).item() > 0:
                print("zzz:nan")
                pass
            # 每一个view都通过qz_inference提取特征
            fea_s = self.qz_inference_s[v](x_list[v])
            fea_p = self.qz_inference_p[v](x_list[v])   
            fea_p_list.append(fea_p)
            fea_s_list.append(fea_s)
        fea_s_list.append(map_fea)
        for fea_s in (fea_s_list):
            z_mu_v_s, z_sca_v_s = self.qz_inference_header(fea_s)
            uniview_mu_s_list.append(z_mu_v_s)
            uniview_sca_s_list.append(z_sca_v_s)
        return uniview_mu_s_list, uniview_sca_s_list, fea_p_list
    
    def generation_x(self, z):
        xr_dist = []
        for v in range(self.num_views):
            xrs_loc = self.px_generation[v](z)
            xr_dist.append(xrs_loc)
        return xr_dist
    
    def poe_aggregate(self, mu, var, mask=None, eps=1e-5):
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0,1).unsqueeze(-1)
        # mask_matrix = torch.stack(mask, dim=0)
        mask_matrix_new = torch.cat([torch.ones([1,mask_matrix.shape[1],mask_matrix.shape[2]]).cuda(),mask_matrix],dim=0)
        p_z_mu = torch.zeros([1,mu.shape[1],mu.shape[2]]).cuda()
        p_z_var = torch.ones([1,mu.shape[1],mu.shape[2]]).cuda()
        mu_new = torch.cat([p_z_mu,mu],dim=0)
        var_new = torch.cat([p_z_var,var],dim=0)
        exist_mu = mu_new * mask_matrix_new
        T = 1. / (var_new+eps)
        if torch.sum(torch.isnan(exist_mu)).item()>0:
            print('.')
        if torch.sum(torch.isinf(T)).item()>0:
            print('.')
        exist_T = T * mask_matrix_new
        aggregate_T = torch.sum(exist_T, dim=0)
        aggregate_var = 1. / (aggregate_T + eps)
        # if torch.sum(torch.isnan(aggregate_var)).item()>0:
        #     print('.')
        aggregate_mu = torch.sum(exist_mu * exist_T, dim=0) / (aggregate_T + eps)
        if torch.sum(torch.isnan(aggregate_mu)).item()>0:
            print(',')
        return aggregate_mu, aggregate_var
    
    def moe_aggregate(self, mu, var, mask=None, eps=1e-5):
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0,1).unsqueeze(-1)
        exist_mu = mu * mask_matrix
        exist_var = var * mask_matrix
        aggregate_var = exist_var.sum(dim=0)
        aggregate_mu = exist_mu.sum(dim=0)
        return aggregate_mu,aggregate_var
    
    def forward(self, x_list, step, map_fea, mask=None):
        # 先将多视图数据x_list 输入inference模块，计算出每一个view的特征的潜在均值和方差
        if (step == 2):
            mu_s_list, sca_s_list, fea_p_list = self.inference_z2(x_list, map_fea)
        z_mu = torch.stack(mu_s_list,dim=0) # [v n d]
        z_sca = torch.stack(sca_s_list,dim=0) # [v n d]
        if torch.sum(torch.isnan(z_mu)).item() > 0:
            print("z:nan")
            pass
        # 通过poe融合技术将先前计算出来的每一个view的均值和方差进行融合
        # 在sip这篇文章中，fusion得到的结果就是共享信息
        fusion_mu, fusion_sca = self.poe_aggregate(z_mu, z_sca, mask=None)
        # 进行重参数化，从融合分布采样潜在变量z_sample
        z_sample = gaussian_reparameterization_var(fusion_mu, fusion_sca,times=10)
        z_sample_list_s = []
        for i in range(len(sca_s_list)):
            z_sample_view_s = gaussian_reparameterization_var(mu_s_list[i], sca_s_list[i], times=5)
            z_sample_list_s.append(z_sample_view_s)
        
        # 使用generation从采样的z中重构视图，结果是一个list，是每个视图的表示
        xr_list = self.generation_x(z_sample)
        xr_p_list = []
        for v in range(self.num_views):
            reconstruct_x_p = self.px_generation_p[v](fea_p_list[v])
            xr_p_list.append(reconstruct_x_p)
        
        # 计算私有特征z_sample_list_p和共享特征z_sample的余弦相似度
        # cos_loss = compute_cosine_similarity(z_sample, z_sample_list_p)
        cos_loss = compute_cosine_similarity_list(z_sample_list_s, fea_p_list)

        return z_sample, mu_s_list, sca_s_list, fusion_mu, fusion_sca, xr_list, xr_p_list, cos_loss, fea_p_list, z_mu, z_sca
