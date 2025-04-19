from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import math,random
import scipy.io
from scipy.io import loadmat
def loadMfDIMvMlDataFromMat(mat_path, fold_mat_path,fold_idx=0):
    # load multiple folds double incomplete multi-view multi-label data and labels
    # mark sure the out dimension is n x d, where n is the number of samples
    data = scipy.io.loadmat(mat_path)
    datafold = scipy.io.loadmat(fold_mat_path)
    mv_data = data['X'][0]
    labels = data['label']
    labels = labels.astype(np.float32)
    index_file = loadmat(fold_mat_path)
    train_indices = index_file['train_indices'].reshape(-1)
    # print(train_indices.shape)
    val_indices = index_file['val_indices'].reshape(-1)
    test_indices = index_file['test_indices'].reshape(-1)
    all_indices = np.concatenate([train_indices, val_indices, test_indices])
    # contains_zero = 0 in all_indices
    # # 输出结果
    # if contains_zero:
    #     print("The index contains 0.")
    # else:
    #     print("The index does not contain 0.")
    mask_train = index_file['mask_train']
    mask_val = index_file['mask_val']
    mask_test = index_file['mask_test']
    combined_mask = np.vstack([mask_train, mask_val, mask_test])
    inc_view_indicator = combined_mask
    train_label_mask = index_file['label_M']
    total_sample_num = labels.shape[0]
    labels=labels[all_indices]
    inc_labels = labels
    inc_label_indicator = train_label_mask
    # print(train_label_mask.shape)
    inc_mv_data = [(StandardScaler().fit_transform(v_data.astype(np.float32))) for
                   v, v_data in enumerate(mv_data)]
    # 假设all_indices已经通过np.concatenate生成
    # 按照all_indices重新排列每个视图中的数据
    inc_mv_data_new = [v_data[all_indices,:] for v_data in inc_mv_data]
    return inc_mv_data_new,inc_labels,labels,inc_view_indicator,inc_label_indicator,total_sample_num,train_indices,val_indices,test_indices
class IncDataset(Dataset):
    def __init__(self,mat_path, fold_mat_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',semisup=False):
        inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind, total_sample_num,train_indices,val_indices,test_indices = loadMfDIMvMlDataFromMat(mat_path,
                                                                                                          fold_mat_path,
                                                                                                          fold_idx)
        self.train_sample_num = len(train_indices)
        # print(self.train_sample_num)
        self.val_sample_num =len(val_indices)
        # print(self.val_sample_num)
        self.test_sample_num = len(test_indices)
        # print(self.test_sample_num)
        if mode == 'train':
            self.cur_mv_data = [v_data[:self.train_sample_num] for v_data in inc_mv_data]
            self.cur_inc_V_ind = inc_V_ind[:self.train_sample_num]
            self.cur_inc_L_ind = inc_L_ind[:self.train_sample_num]
            self.cur_labels = inc_labels[:self.train_sample_num]* self.cur_inc_L_ind
        elif mode == 'val':
            self.cur_mv_data = [v_data[self.train_sample_num:self.train_sample_num + self.val_sample_num] for v_data in
                                inc_mv_data]
            self.cur_labels = labels[self.train_sample_num:self.train_sample_num + self.val_sample_num]
            self.cur_inc_V_ind = inc_V_ind[self.train_sample_num:self.train_sample_num + self.val_sample_num]
            self.cur_inc_L_ind = np.ones_like(self.cur_labels)
            # print('self.cur_inc_V_ind=', self.cur_inc_V_ind)
            # print('self.cur_inc_L_ind=', self.cur_inc_L_ind)
        else:
            self.cur_mv_data = [v_data[self.train_sample_num + self.val_sample_num:] for v_data in inc_mv_data]
            self.cur_labels = labels[self.train_sample_num + self.val_sample_num:]
            self.cur_qinc_V_ind = inc_V_ind[self.train_sample_num + self.val_sample_num:]
            self.cur_inc_L_ind = np.ones_like(self.cur_labels)
        self.mode = mode
        self.classes_num = labels.shape[1]
        self.d_list = [da.shape[1] for da in inc_mv_data]
        self.view_num=len(inc_mv_data)
    def __len__(self):
        if self.mode == 'train':
            return self.train_sample_num
        elif self.mode == 'val':
            return self.val_sample_num
        else: return self.test_sample_num
    def __getitem__(self, index):
        # index = index if self.is_train else self.train_sample_num+index
        data = [torch.tensor(v[index],dtype=torch.float) for v in self.cur_mv_data]
        label = torch.tensor(self.cur_labels[index], dtype=torch.float)
        inc_V_ind = torch.tensor(self.cur_inc_V_ind[index], dtype=torch.int32)
        inc_L_ind = torch.tensor(self.cur_inc_L_ind[index], dtype=torch.int32)
        return data,label,inc_V_ind,inc_L_ind
def getIncDataloader(matdata_path, fold_matdata_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',batch_size=1,num_workers=1,shuffle=False):
    dataset = IncDataset(matdata_path, fold_matdata_path, training_ratio=training_ratio, val_ratio=val_ratio, mode=mode, fold_idx=fold_idx)
    dataloder = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    # for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(dataloder):
    #     print(i,'=',inc_V_ind)
    return dataloder,dataset

def loadDirectMvMlDataFromMat(mat_path, fold_idx=0, training_ratio=0.7, val_ratio=0.15):
    """
    直接从原始.mat文件加载多视图多标签数据，但只选择两个视图
    """
    # 加载.mat文件
    data = scipy.io.loadmat(mat_path)
    
    # 使用与原始代码相同的方式获取数据
    if 'X' in data:
        all_views = data['X'][0]  # 获取所有视图
    elif 'features' in data:
        all_views = data['features'][0]  # 获取所有视图
    else:
        raise ValueError("mat文件中既没有'X'也没有'features'")
    
    # 打印视图数量信息
    print(f"原始视图数量: {len(all_views)}")
    
    # 随机选择两个视图
    np.random.seed(fold_idx)
    if len(all_views) >= 2:
        # 如果有至少两个视图，随机选择两个
        selected_indices = np.random.choice(len(all_views), 2, replace=False)
        print(f"选择的视图索引: {selected_indices}")
        
        # 只保留选择的两个视图
        mv_data = [all_views[i] for i in selected_indices]
    else:
        # 如果视图数量小于2，使用所有可用视图并打印警告
        print("警告: 视图数量少于2，使用所有可用视图")
        mv_data = all_views
    
    labels = data['label']  # 获取标签
    labels = labels.astype(np.float32)
    
    # 打印数据信息
    print(f"标签形状: {labels.shape}")
    print(f"选择的视图数量: {len(mv_data)}")
    for i, v in enumerate(mv_data):
        print(f"视图 {i} 形状: {v.shape if hasattr(v, 'shape') else '未知'}")
    
    # 获取样本总数
    total_sample_num = labels.shape[0]
    
    # 生成训练/验证/测试分割
    np.random.seed(fold_idx)
    indices = np.random.permutation(total_sample_num)
    
    train_size = int(total_sample_num * training_ratio)
    val_size = int(total_sample_num * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 创建视图指示器 - 现在只有两个视图
    view_num = len(mv_data)
    inc_view_indicator = np.ones((total_sample_num, view_num))
    
    # 创建标签指示器
    inc_label_indicator = np.ones_like(labels)
    
    # 标准化特征
    inc_mv_data = []
    for v_data in mv_data:
        if not isinstance(v_data, np.ndarray) or v_data.dtype.kind not in 'fiub':
            print(f"警告: 视图数据不是数值数组，类型: {type(v_data)}, dtype: {v_data.dtype if hasattr(v_data, 'dtype') else '未知'}")
            # 创建一个替代特征矩阵
            v_data = np.ones((total_sample_num, 10))
        
        # 确保视图数据的行数与样本数匹配
        if v_data.shape[0] != total_sample_num:
            print(f"警告: 视图形状 {v_data.shape} 与样本数 {total_sample_num} 不匹配")
            # 如果视图数据行数少于样本数，我们可以填充或截断
            if v_data.shape[0] < total_sample_num:
                # 填充到需要的大小
                padding = np.zeros((total_sample_num - v_data.shape[0], v_data.shape[1]))
                v_data = np.vstack([v_data, padding])
            else:
                # 截断到需要的大小
                v_data = v_data[:total_sample_num]
        
        # 标准化特征
        try:
            standardized_data = StandardScaler().fit_transform(v_data)
            inc_mv_data.append(standardized_data)
        except Exception as e:
            print(f"标准化失败: {e}")
            # 如果标准化失败，使用原始数据
            inc_mv_data.append(v_data)
    
    return inc_mv_data, labels, labels, inc_view_indicator, inc_label_indicator, total_sample_num, train_indices, val_indices, test_indices

class DirectDataset(Dataset):
    def __init__(self, mat_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train'):
        inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind, total_sample_num, train_indices, val_indices, test_indices = loadDirectMvMlDataFromMat(
            mat_path, fold_idx, training_ratio, val_ratio)
        
        self.train_sample_num = len(train_indices)
        self.val_sample_num = len(val_indices)
        self.test_sample_num = len(test_indices)
        
        # 选择当前模式的数据
        if mode == 'train':
            self.cur_indices = train_indices
        elif mode == 'val':
            self.cur_indices = val_indices
        else:  # test
            self.cur_indices = test_indices
        
        # 确保所有视图数据都能被索引访问
        self.cur_mv_data = []
        for v_data in inc_mv_data:
            try:
                # 使用当前模式的索引获取数据子集
                if len(self.cur_indices) > 0:
                    # 确保索引不超出范围
                    valid_indices = [idx for idx in self.cur_indices if idx < v_data.shape[0]]
                    if len(valid_indices) < len(self.cur_indices):
                        print(f"警告: 舍弃了 {len(self.cur_indices) - len(valid_indices)} 个无效索引")
                    
                    if len(valid_indices) > 0:
                        self.cur_mv_data.append(v_data[valid_indices])
                    else:
                        # 如果没有有效索引，创建一个空数组
                        print("警告: 没有有效索引，创建空数组")
                        self.cur_mv_data.append(np.zeros((1, v_data.shape[1])))
                else:
                    # 如果没有索引，创建一个空数组
                    self.cur_mv_data.append(np.zeros((1, v_data.shape[1])))
            except Exception as e:
                print(f"处理视图数据时出错: {e}")
                # 创建一个空数组作为替代
                self.cur_mv_data.append(np.zeros((len(self.cur_indices) if len(self.cur_indices) > 0 else 1, 10)))
        
        # 处理标签和指示器
        if len(self.cur_indices) > 0:
            # 确保索引不超出范围
            valid_indices = [idx for idx in self.cur_indices if idx < labels.shape[0]]
            if len(valid_indices) < len(self.cur_indices):
                print(f"警告: 处理标签时舍弃了 {len(self.cur_indices) - len(valid_indices)} 个无效索引")
            
            if len(valid_indices) > 0:
                self.cur_labels = labels[valid_indices]
                self.cur_inc_V_ind = inc_V_ind[valid_indices]
                self.cur_inc_L_ind = inc_L_ind[valid_indices]
            else:
                # 如果没有有效索引，创建空数组
                self.cur_labels = np.zeros((1, labels.shape[1]))
                self.cur_inc_V_ind = np.ones((1, len(inc_mv_data)))
                self.cur_inc_L_ind = np.ones((1, labels.shape[1]))
        else:
            # 如果没有索引，创建空数组
            self.cur_labels = np.zeros((1, labels.shape[1]))
            self.cur_inc_V_ind = np.ones((1, len(inc_mv_data)))
            self.cur_inc_L_ind = np.ones((1, labels.shape[1]))
        
        self.mode = mode
        self.classes_num = labels.shape[1]
        self.d_list = [da.shape[1] for da in self.cur_mv_data]
        self.view_num = len(self.cur_mv_data)
    
    def __len__(self):
        return len(self.cur_labels)
    
    def __getitem__(self, index):
        try:
            data = [torch.tensor(v[index], dtype=torch.float) for v in self.cur_mv_data]
            label = torch.tensor(self.cur_labels[index], dtype=torch.float)
            inc_V_ind = torch.tensor(self.cur_inc_V_ind[index], dtype=torch.int32)
            inc_L_ind = torch.tensor(self.cur_inc_L_ind[index], dtype=torch.int32)
            return data, label, inc_V_ind, inc_L_ind
        except Exception as e:
            print(f"获取数据项 {index} 时出错: {e}")
            # 返回零填充的数据
            dummy_data = [torch.zeros(1, d) for d in self.d_list]
            dummy_label = torch.zeros(1, self.classes_num)
            dummy_v_ind = torch.ones(1, self.view_num, dtype=torch.int32)
            dummy_l_ind = torch.ones(1, self.classes_num, dtype=torch.int32)
            return dummy_data, dummy_label, dummy_v_ind, dummy_l_ind

def getDirectDataloader(matdata_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train', batch_size=1, num_workers=1, shuffle=False):
    dataset = DirectDataset(matdata_path, training_ratio=training_ratio, val_ratio=val_ratio, mode=mode, fold_idx=fold_idx)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, dataset

if __name__=='__main__':
    # dataloder,dataset = getComDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    dataloder,dataset = getIncDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat','/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    labels = torch.tensor(dataset.cur_labels).float()
