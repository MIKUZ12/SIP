import torch
import torch.nn as nn 
# from Layers import EncoderLayer, DecoderLayer
# from Embed import Embedder, PositionalEncoder
import random
import copy
import math
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
from layers_new import GIN, FDModel, MLP,GAT
from model_VAE_new import VAE
def gaussian_reparameterization_std(means, std, times=1):
    std = std.abs()
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times

def gaussian_reparameterization_var(means, var, times=1):
    ## 重参数化
    std = torch.sqrt(var+1e-8)
    assert torch.sum(std<0).item()==0
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times

def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    

def fill_with_label(label_embedding,label,x_embedding,inc_V_ind):
    fea = label.matmul(label_embedding)
    new_x =  x_embedding*inc_V_ind.unsqueeze(-1) + fea.unsqueeze(1)
    return new_x

class MLP(nn.Module):
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,final_act=True,final_norm=True):
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

class Qc_inference_mlp(nn.Module):
    # 变分推断部分--encoder，将输入数据映射为高斯分布的参数： 均值mu和尺度sca
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(Qc_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())
        # self.qzv_inference = nn.Sequential(*self.qz_layer)
    def forward(self, x):
        assert torch.sum(torch.isnan(x)).item() == 0
        hidden_features = self.mlp(x)
        c_mu = self.z_loc(hidden_features)
        c_sca = self.z_sca(hidden_features)
        # class_feature  = self.z
        if torch.sum(torch.isnan(c_mu)).item() >0:
            pass
        assert torch.sum(torch.isinf(c_mu)).item() == 0
        return c_mu, c_sca

class Net(nn.Module):
    def __init__(self, d_list,num_classes,z_dim,adj,rand_seed=0):
        super(Net, self).__init__()
        # self.configs = configs
        self.rand_seed = rand_seed
        # Label semantic encoding module
        self.label_embedding_u = nn.Parameter(torch.eye(num_classes),
                                            requires_grad=True)
        self.mlp_2view = MLP(d_list[0], d_list[1])
        # self.label_embedding = nn.Parameter(torch.randn([num_classes,num_classes]),
        #                                     requires_grad=True)
        self.label_adj = nn.Parameter(torch.eye(num_classes),
                                      requires_grad=True)
        self.adj = adj
        # z_dim 是什么？？
        self.z_dim = z_dim
        # self.GIN_encoder = GIN(2, num_classes, z_dim,
        #                        [math.ceil(z_dim / 2)])
        self.GAT_encoder = GAT(z_dim, z_dim,dropout=0.0)
        self.label_mlp = Qc_inference_mlp(num_classes, z_dim)
        self.mix_prior = None
        self.mix_mu = None
        self.mix_sca = None
        self.k = num_classes
        # VAE module
        self.VAE = VAE(d_list=d_list,z_dim=z_dim,class_num=num_classes)

        # Classifier
        self.cls_conv = nn.Conv1d(num_classes, num_classes,
                                  z_dim*2, groups=num_classes)
        
        self.cls = nn.Linear(z_dim, num_classes)
        self.view_cls = nn.ModuleList([nn.Linear(z_dim, num_classes) for i in range(len(d_list)) ])
        self.set_prior()
        self.cuda()
        # self.reset_parameters()
    def set_prior(self):
        # set prior components weights
        # 为混合高斯分布初始化参数
        self.mix_prior = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)
        # set prior gaussian components mean
        # self.mix_mu = nn.Parameter(torch.full((self.k, self.z_dim), 0.0), requires_grad=True)
        self.mix_mu = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)
        # set prior gaussian components scale
        self.mix_sca = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)
        
    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        # nn.init.normal_(self.label_embedding_u)
        nn.init.normal_(self.label_adj)
        # self.GIN_encoder.reset_parameters()
        self.FD_model.reset_parameters()
        self.cls_conv.reset_parameters()
        
    def get_config_optim(self):
        return [{'params': self.GAT_encoder.parameters()},
                {'params': self.FD_model.parameters()},
                {'params': self.cls_conv.parameters()}]
        
    def poe_two(self, z_mu, z_var, c_mu, c_var, eps=1e-5):
        # POE fusion
        z_mu = z_mu.unsqueeze(1)   # [B, 1, D]
        z_var = z_var.unsqueeze(1) # [B, 1, D]
        c_mu = c_mu.unsqueeze(0)   # [1, K, D]
        c_var = c_var.unsqueeze(0) # [1, K ,D]
        # p_z_var = torch.ones([1,mu.shape[1],mu.shape[2]]).cuda()
        # mu_new = torch.cat([p_z_mu,mu],dim=0)
        # var_new = torch.cat([p_z_var,var],dim=0)
        # 此处是模拟一个很广泛的分布，防止其他专家过度主导融合结果
        s_mu = torch.zeros([z_mu.shape[0],c_mu.shape[1],z_mu.shape[2]]).cuda() # [B, K, D]
        s_var = torch.ones([z_mu.shape[0],c_mu.shape[1],z_mu.shape[2]]).cuda() # [B, K, D]
        # mu_new = torch.cat([p_z_mu,mu],dim=0)
        # var_new = torch.cat([p_z_var,var],dim=0)
        # 这是计算精度矩阵（逆方差）
        T_x = 1. / (z_var+eps)
        T_c = 1. / (c_var+eps)
        T_s = 1. / (s_var+eps)
        # 融合均值和方差
        T_sum = T_x + T_c + T_s # 总精度
        aggregate_mu = (z_mu*T_x+c_mu*T_c+ s_mu*T_s)/T_sum
        aggregate_var = 1. / T_sum
        # mask_matrix = torch.stack(mask, dim=0)
        if torch.sum(torch.isnan(aggregate_mu)).item()>0:
            print(',')
        assert torch.sum(torch.isnan(aggregate_mu)).item()==0
        assert torch.sum(torch.isinf(aggregate_mu)).item()==0
        assert torch.sum(torch.isnan(aggregate_var)).item()==0
        assert torch.sum(torch.isinf(aggregate_var)).item()==0
        return aggregate_mu, aggregate_var
    
    def forward(self, x_list, mask):
        # Generating semantic label embeddings via label semantic encoding module
        # label_embedding = self.GIN_encoder(self.label_embedding, self.label_adj)
        # print(self.label_adj[:10,:10])
        label_embedding  =  self.label_embedding_u
        # 这里还需要修改，此处是默认取了第一个视图作为保留视图(索引0)
        mapped_data = self.mlp_2view(x_list[0])
        # 用 mapped_data 替代后续的消失视图
        # x_list[1] = mapped_data
        # label_embedding = gaussian_reparameterization_std(self.label_embedding_u,self.label_embedding_std)
        # label_embedding = self.GAT_encoder(label_embedding, self.adj)
        assert torch.sum(torch.isnan(self.label_embedding_u)).item() == 0
        assert torch.sum(torch.isinf(self.label_embedding_u)).item() == 0
        # 将标签的嵌入label_embedding_u（初始化为一个大小为num_classes*num_classes的对角矩阵）输入进变分推断的encoder，得到高斯分布的均值和幅度sca
        label_embedding, label_embedding_var = self.label_mlp(self.label_embedding_u)
        # 做一次重参数化，保证可微
        label_embedding_sample = gaussian_reparameterization_var(label_embedding,label_embedding_var,10)
        assert torch.sum(torch.isnan(label_embedding_sample)).item() == 0
        assert torch.sum(torch.isinf(label_embedding_sample)).item() == 0
        # label_embedding_sample = self.GAT_encoder(label_embedding_sample, self.adj)
        if torch.sum(torch.isnan(label_embedding)).item() > 0:
            assert torch.sum(torch.isnan(label_embedding)).item() == 0
        # x_list是输入的data，mask是掩码
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, xr_p_list, cos_loss, z_sample_list_p = self.VAE(x_list,mask)  #Z[i]=[128, 260, 512] b c d_e
        if torch.sum(torch.isnan(z_sample)).item() > 0:
            pass
        # z_sample是从编码器得到的混合分布中采样的潜在变量，label_embedding_sample是标签嵌入之后采样的潜在标签，然后连接起来
        qc_z = torch.cat((z_sample.unsqueeze(1).repeat(1,label_embedding_sample.shape[0],1),label_embedding_sample.unsqueeze(0).repeat(z_sample.shape[0],1,1)),dim=-1)
        # 连接起来的潜在特征经过一个fully connected layer
        p_s = self.cls_conv(qc_z).squeeze(-1)
        qc_p0 = torch.cat((z_sample_list_p[0].unsqueeze(1).repeat(1,label_embedding_sample.shape[0],1),label_embedding_sample.unsqueeze(0).repeat(z_sample_list_p[0].shape[0],1,1)),dim=-1)
        qc_p1 = torch.cat((z_sample_list_p[1].unsqueeze(1).repeat(1,label_embedding_sample.shape[0],1),label_embedding_sample.unsqueeze(0).repeat(z_sample_list_p[1].shape[0],1,1)),dim=-1)     
        p_p0 = self.cls_conv(qc_p0).squeeze(-1)
        p_p1 = self.cls_conv(qc_p1).squeeze(-1)
        # 再经过一次sigmoid，以上三步对应的是论文（公式11）
        p_s = torch.sigmoid(p_s)
        p_p0 = torch.sigmoid(p_p0)
        p_p1 = torch.sigmoid(p_p1)

        # 计算每个分类器的交叉熵损失
        loss_s = - (p_s * torch.log(p_s) + (1 - p_s) * torch.log(1 - p_s))
        loss_p0 = - (p_p0 * torch.log(p_p0) + (1 - p_p0) * torch.log(1 - p_p0))
        loss_p1 = - (p_p1 * torch.log(p_p1) + (1 - p_p1) * torch.log(1 - p_p1))

        # 计算损失的倒数作为权重
        weight_s = 1 / loss_s
        weight_p0 = 1 / loss_p0
        weight_p1 = 1 / loss_p1

        # 合并权重并归一化
        weights = torch.stack([weight_s, weight_p0, weight_p1])
        weights_normalized = weights / weights.sum()

        # 对三个分类器的预测结果进行加权融合
        p_fused = weights_normalized[0] * p_s + weights_normalized[1] * p_p0 + weights_normalized[2] * p_p1
        
        return z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, label_embedding_sample, p_fused, xr_p_list, cos_loss, mapped_data

def get_model(d_list,num_classes,z_dim,adj,rand_seed=0):
    model = Net(d_list,num_classes=num_classes,z_dim=z_dim,adj=adj,rand_seed=rand_seed)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() 
                                    else 'cpu'))
    return model
    
if __name__=="__main__":
    # input=torch.ones([2,10,768])
    from MLdataset import getIncDataloader
    dataloder,dataset = getIncDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat','/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0_LabelMaskRatio_0_TraindataRatio_0.7.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    input = next(iter(dataloder))[0]
    model=get_model(num_classes=260,beta=0.2,in_features=1,class_emb=260,rand_seed=0)
    input = [v_data.to('cuda:0') for v_data in input]
    # print(input[0])
    pred,_,_=model(input)