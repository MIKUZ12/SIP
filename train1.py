import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import utils
from utils import AverageMeter
import torch.nn.functional as F
import MLdataset
import argparse
import time
from model_new import get_model
import evaluation
import torch
import numpy as np
import copy
from myloss import Loss, vade_trick
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
# from constructGraph import getMvKNNGraph, getIncMvKNNGraph
import time
import multiprocessing

def view_mixup(data_list,label,v_mask,l_mask):
    new_data = []
    new_labels = label.unsqueeze(0).repeat(len(data_list),1,1)
    num = data_list[0].shape[0]
    for i,view_data in enumerate(data_list):
        indices = torch.randperm(num).cuda()
        new_data.append(torch.index_select(view_data, 0, indices))
        new_labels[i] = torch.index_select(new_labels[i],0,indices)
        v_mask[:,i] = torch.index_select(v_mask[:,i],0,indices)
    label = new_labels.sum(dim=0)
    label = torch.masked_fill(label,label>0,1)
    l_mask = torch.masked_fill(l_mask,label>0,1)
    return new_data,label,v_mask,l_mask


def train_step1(loader, model, loss_model, opt, sche, epoch,dep_graph,last_preds,logger, selected_view):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mce= nn.MultiLabelSoftMarginLoss()
    model.train()
    end = time.time()
    All_preds = torch.tensor([]).cuda()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)

        data=[v_data.to('cuda:0') for v_data in data]
        original_data = copy.deepcopy(data) 
        original_data = [v_data.to('cuda:0') for v_data in original_data]
        data_selected = [data[i] for i in selected_view]
        # 在两个视图的数据之间构建一个mlp用于映射
        label = label.to('cuda:0')
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')
        # data是多视图的信息
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, label_emb_sample, pred, xr_p_list, cos_loss, mapped_data = model(data_selected, mask=inc_V_ind)
        All_preds = torch.cat([All_preds,pred],dim=0) # 融合之后的预测输出

        if epoch<args.pre_epochs:
            loss_mse_viewspec = 0
            loss_CL_views = 0
            loss_list=[]
            loss = loss_CL_views
            assert torch.sum(torch.isnan(loss)).item() == 0
            assert torch.sum(torch.isinf(loss)).item() == 0
        else:
            loss_CL = loss_model.weighted_BCE_loss(pred,label,inc_L_ind) # 分类损失
            z_c_loss = loss_model.z_c_loss_new(z_sample, label, label_emb_sample,inc_L_ind)
            cohr_loss = loss_model.corherent_loss(uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca,mask=inc_V_ind)
            mapping_loss = F.mse_loss(mapped_data, data_selected[1])
            loss_mse = 0
            loss_mse_p = 0
            loss_pos_beta_I = 0
            for v in range(len(data_selected)):
                ## xr_list是每个重构的视图，这个损失是用来约束VAE的，使得潜在空间的特征能够很好的表征原数据
                ## 对应（公式7）的第一项，最大化重构概率，对应最大化互信息的下界
                loss_mse += loss_model.weighted_wmse_loss(data_selected[v],xr_list[v],inc_V_ind[:,v],reduction='mean')
            for v in range(len(data_selected)):
                ## xr_list是每个重构的视图，这个损失是用来约束VAE的，使得潜在空间的特征能够很好的表征原数据
                loss_mse_p += loss_model.weighted_wmse_loss(data_selected[v],xr_p_list[v],inc_V_ind[:,v],reduction='mean')
            cos_loss = cos_loss
            assert torch.sum(torch.isnan(loss_mse)).item() == 0
            # loss = loss_CL + loss_mse *args.alpha + z_c_loss*args.beta + cohr_loss *args.sigma + loss_mse_p + 0.01*loss_pos_beta_I + mapping_loss
            loss = loss_CL + loss_mse + mapping_loss + cohr_loss + cos_loss*0.1
        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        opt.step()
        # print(model.classifier.parameters().grad)
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))
    return losses,model,All_preds

def train_step2(loader, model, model_last, loss_model, opt, sche, epoch,dep_graph,last_preds,logger, selected_view):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mce= nn.MultiLabelSoftMarginLoss()
    model = model.to('cuda:0')
    model_last = model_last.to('cuda:0')
    model.train()
    model_last.eval()
    end = time.time()
    All_preds = torch.tensor([]).cuda()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]
        complete_view = model_last.mlp_2view(data[selected_view[1]])
        original_data = copy.deepcopy(data) 
        original_data = [v_data.to('cuda:0') for v_data in original_data]
        data_selected = [data[i] for i in selected_view]
        data_selected.append(complete_view)
        # 在两个视图的数据之间构建一个mlp用于映射
        label = label.to('cuda:0')
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')
        # data是多视图的信息
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, label_emb_sample, pred, xr_p_list, pos_beta_I, mapped_data = model(data_selected, mask=inc_V_ind)
        All_preds = torch.cat([All_preds,pred],dim=0) # 融合之后的预测输出

        if epoch<args.pre_epochs:
            loss_mse_viewspec = 0
            loss_CL_views = 0
            loss_list=[]
            loss = loss_CL_views
            assert torch.sum(torch.isnan(loss)).item() == 0
            assert torch.sum(torch.isinf(loss)).item() == 0
        else:
            loss_CL = loss_model.weighted_BCE_loss(pred,label,inc_L_ind) # 分类损失
            z_c_loss = loss_model.z_c_loss_new(z_sample, label, label_emb_sample,inc_L_ind)
            cohr_loss = loss_model.corherent_loss(uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca,mask=inc_V_ind)
            mapping_loss = F.mse_loss(mapped_data, original_data[1])
            loss_mse = 0
            loss_mse_p = 0
            loss_pos_beta_I = 0
            for v in range(len(data_selected)-1):
                ## xr_list是每个重构的视图，这个损失是用来约束VAE的，使得潜在空间的特征能够很好的表征原数据
                ## 对应（公式7）的第一项，最大化重构概率，对应最大化互信息的下界
                loss_mse += loss_model.weighted_wmse_loss(data_selected[v],xr_list[v],inc_V_ind[:,v],reduction='mean')
            for v in range(len(data_selected)-1):
                ## xr_list是每个重构的视图，这个损失是用来约束VAE的，使得潜在空间的特征能够很好的表征原数据
                loss_mse_p += loss_model.weighted_wmse_loss(data_selected[v],xr_p_list[v],inc_V_ind[:,v],reduction='mean')
            loss_pos_beta_I = pos_beta_I
            assert torch.sum(torch.isnan(loss_mse)).item() == 0
            # loss = loss_CL + loss_mse *args.alpha + z_c_loss*args.beta + cohr_loss *args.sigma + loss_mse_p + 0.01*loss_pos_beta_I + mapping_loss
            loss = loss_CL + loss_mse + mapping_loss + cohr_loss
            # 打印出loss_pos_beta_I的大小：
        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        opt.step()
        # print(model.classifier.parameters().grad)
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))
    return losses,model,All_preds



def test(loader, model, loss_model, epoch, logger, selected_view):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        # data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]
        data_selected = [data[i] for i in selected_view]
        # pred,_,_ = model(data,mask=torch.ones_like(inc_V_ind).to('cuda:0'))
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, label_emb_sample, qc, xr_p_list, pos_beta_I, mapped_data = model(data_selected,mask=inc_V_ind.to('cuda:0'))
        # qc_x = vade_trick(fusion_z_mu, model.mix_prior, model.mix_mu, model.mix_sca)
        pred = qc
        pred = pred.cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()
        loss=loss_model.weighted_BCE_loss(pred,label,inc_L_ind)
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)
    evaluation_results=evaluation.do_metric(total_preds,total_labels)
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   batch_time=batch_time,
                        losses = losses,
                        ap=evaluation_results[4],
                        hl=evaluation_results[0],
                        rl=evaluation_results[3],
                        auc=evaluation_results[5]
                        ))
    return evaluation_results

def main(args,file_path):
    panduan_list=[0]
    for panduan in panduan_list:
        if panduan==0:
            args.label_missing_rates=[0]
            args.feature_missing_rates=[0]
            args.folds_num=3
        for label_missing_rate in args.label_missing_rates:
            for feature_missing_rate in args.feature_missing_rates:
                folds_num = args.folds_num
                test_acc_list = []
                test_one_error_list = []
                test_coverage_list = []
                test_ranking_loss_list = []
                test_precision_list = []
                test_auc_list = []
                test_f1_list = []
                training_time_list = []
                for fold_idx in range(folds_num):
                    seed=fold_idx
                    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'.mat')
                    # fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset + str(
                    #     mask_label_ratio) + '_LabelMaskRatio_' + str(mask_view_ratio) + '_MaskRatios_' + '.mat')
                    dataset_name = args.dataset
                    folder_name = f'E:/desktop/SIP/Index/{dataset_name}'
                    file_name = f'{dataset_name}_label_missing_rate_{label_missing_rate}_feature_missing_rate_{feature_missing_rate}_seed_{seed + 1}.mat'
                    data_path = osp.join(args.root_dir, args.dataset + '.mat')
                    index_file_path = os.path.join(folder_name, file_name)
                    fold_data_path=index_file_path
                    folds_results = [AverageMeter() for i in range(9)]
                    if args.logs:
                        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                                    args.mask_view_ratio) + '_L_' +
                                                    str(args.mask_label_ratio) + '_T_' +
                                                    str(args.training_sample_ratio) + '_'+str(args.alpha)+'_'+str(args.beta)+'.txt')
                    else:
                        logfile=None
                    logger = utils.setLogger(logfile)
                    device = torch.device('cuda:0')
                    label_Inp_list = []
                    fold_idx=fold_idx
                    train_dataloder,train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='train',batch_size=args.batch_size,shuffle = True,num_workers=0)
                    test_dataloder,test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.15,fold_idx=fold_idx,mode='test',batch_size=args.batch_size,num_workers=0)
                    val_dataloder,val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='val',batch_size=args.batch_size,num_workers=0)
                    d_list = train_dataset.d_list
                    classes_num = train_dataset.classes_num
                    labels = torch.tensor(train_dataset.cur_labels).float().to('cuda:0')
                    # 直到这里前面都是数据处理的部分，和之前的代码如出一辙
            
                    # 计算标签矩阵的相关性矩阵C
                    dep_graph = torch.matmul(labels.T,labels)
                    dep_graph = dep_graph/(torch.diag(dep_graph).unsqueeze(1)+1e-10)
                    # dep_graph[dep_graph<=args.sigma]=0.
                    dep_graph.fill_diagonal_(fill_value=0.)

                    start = time.time()
                    model=get_model(d_list,num_classes=classes_num,z_dim=args.z_dim,adj=dep_graph,rand_seed=0)
                    loss_model = Loss()
                    optimizer = Adam(model.parameters(), lr=args.lr)
                    scheduler = None
                    logger.info('train_data_num:'+str(len(train_dataset))+'  test_data_num:'+str(len(test_dataset))+'   fold_idx:'+str(fold_idx))
                    print(args)
                    static_res = 0
                    epoch_results = [AverageMeter() for i in range(9)]
                    total_losses = AverageMeter()
                    train_losses_last = AverageMeter()
                    best_epoch=0
                    best_model_dict = {'model':model.state_dict(),'epoch':0}

                    # 设置第一阶段和第二阶段的分界线
                    bound = 10

                    for epoch in range(args.epochs):
                        if (epoch <= bound):
                            selected_view = [1, 0]
                            print("selected_view: ",selected_view)
                            d_list_selected = [train_dataset.d_list[i] for i in selected_view]
                            model_1 = get_model(d_list_selected,num_classes=classes_num,z_dim=args.z_dim,adj=dep_graph,rand_seed=0)
                            optimizer_1 = Adam(model_1.parameters(), lr=args.lr)
                            if epoch==0:
                                All_preds = None
                            train_losses,model,All_preds = train_step1(train_dataloder,model_1,loss_model,optimizer_1,scheduler,epoch,dep_graph,All_preds,logger, selected_view)
  
                            if epoch >= args.pre_epochs:

                                val_results = test(val_dataloder,model,loss_model,epoch,logger, selected_view)
                                if val_results[0]*0.25+val_results[4]*0.25+val_results[5]*0.25>=static_res:
                                    static_res = val_results[0]*0.25+val_results[4]*0.25+val_results[5]*0.25
                                    best_model_dict['model'] = copy.deepcopy(model.state_dict())
                                    best_model_dict['epoch'] = epoch
                                    best_epoch=epoch
                                train_losses_last = train_losses
                                total_losses.update(train_losses.sum)
                        else:
                            print("进入第二阶段")
                            selected_view = [2, 1]
                            print("selected_view: ",selected_view)
                            d_list_selected = [train_dataset.d_list[i] for i in selected_view]
                            model_2 = get_model(d_list_selected,num_classes=classes_num,z_dim=args.z_dim,adj=dep_graph,rand_seed=0)
                            model_2 = model_2.to('cuda:0')
                            optimizer_2 = Adam(model_2.parameters(), lr=args.lr)
                            if epoch==0:
                                All_preds = None
                            train_losses,model,All_preds = train_step2(train_dataloder,model_2, model_1, loss_model,optimizer_2,scheduler,epoch,dep_graph,All_preds,logger, selected_view)
                            
                            if epoch >= args.pre_epochs:

                                val_results = test(val_dataloder,model,loss_model,epoch,logger, selected_view)
                                if val_results[0]*0.25+val_results[4]*0.25+val_results[5]*0.25>=static_res:
                                    static_res = val_results[0]*0.25+val_results[4]*0.25+val_results[5]*0.25
                                    best_model_dict['model'] = copy.deepcopy(model.state_dict())
                                    best_model_dict['epoch'] = epoch
                                    best_epoch=epoch
                                train_losses_last = train_losses
                                total_losses.update(train_losses.sum)
                            
                    model.load_state_dict(best_model_dict['model'])
                    end = time.time()
                    print("epoch",best_model_dict['epoch'])
                    test_results = test(test_dataloder,model,loss_model,epoch,logger, selected_view)
                    test_acc_list.append(test_results[0])
                    test_one_error_list.append(test_results[1])
                    test_coverage_list.append(test_results[2])
                    test_ranking_loss_list.append(test_results[3])
                    test_precision_list.append(test_results[4])
                    test_auc_list.append(test_results[5])
                    test_f1_list.append(test_results[6])
                    training_time_list.append(end - start)
                if 1 + 1 == 2:
                    # 修改保存路径的部分
                    saved_path = f"./test_results/{dataset_name}/"
                    if not os.path.exists(saved_path):
                        os.makedirs(saved_path)
                    # 修改后的文件名为 missing_rate_i{missing_rate_i}_positive_noise_rate_{positive_noise_rate}_negative_noise_rate_{negative_noise_rate}.txt
                    saved_path = saved_path + f"_gamma_{args.gamma}_alpha_{args.alpha}_laebl_missing_rate_{label_missing_rate}_feature_missing_rate_{feature_missing_rate}.txt"
                    with open(saved_path, 'w') as file:
                        file.write(
                            '============================ Summary Board ============================\n')
                        file.write('>>>>>>>> Experimental Setup:\n')
                        file.write('>>>>>>>> Dataset Info:\n')
                        file.write(f'Dataset name: {dataset_name}\n')
                        file.write('>>>>>>>> Experiment Results:\n')
                        for i in range(1, len(test_acc_list) + 1):
                            file.write(f'Test{i}:  ')
                            file.write(
                                'Accuracy:{0:.4f} | one_error:{1:.4f}| coverage:{2:.4f} | ranking_loss:{3:.4f}| Precision:{4:.4f} | AUC:{5:.4f} | F1:{6:.4f} | \n'.format(
                                    test_acc_list[i - 1], test_one_error_list[i - 1],
                                    test_coverage_list[i - 1],
                                    test_ranking_loss_list[i - 1], test_precision_list[i - 1],
                                    test_auc_list[i - 1], test_f1_list[i - 1]))
                        file.write('>>>>>>>> Mean Score (standard deviation):\n')
                        file.write(
                            'Accuracy: {0:.4f} ({1:.4f})\n'.format(np.mean(test_acc_list),
                                                                   np.std(test_acc_list)))
                        file.write('one_error: {0:.4f} ({1:.4f})\n'.format(
                            np.mean(test_one_error_list),
                            np.std(test_one_error_list)))
                        file.write(
                            'coverage: {0:.4f} ({1:.4f})\n'.format(np.mean(test_coverage_list),
                                                                   np.std(test_coverage_list)))
                        file.write('ranking_loss: {0:.4f} ({1:.4f})\n'.format(
                            np.mean(test_ranking_loss_list),
                            np.std(test_ranking_loss_list)))
                        file.write('Precision: {0:.4f} ({1:.4f})\n'.format(
                            np.mean(test_precision_list),
                            np.std(test_precision_list)))
                        file.write(
                            'AUC: {0:.4f} ({1:.4f})\n'.format(np.mean(test_auc_list),
                                                              np.std(test_auc_list)))
                        file.write(
                            'F1: {0:.4f} ({1:.4f})\n'.format(np.mean(test_f1_list),
                                                             np.std(test_f1_list)))
                        file.write(
                            '>>>>>>>> Mean Training Time (standard deviation)(s):{0:.4f} ({1:.4f})\n'.format(
                                np.mean(training_time_list), np.std(training_time_list)))
                        for itr in range(len(training_time_list)):
                            file.write(f"{itr}: {training_time_list[itr]}\n")
                    file.close()

def filterparam(file_path,index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'final_records'))
    parser.add_argument('--file-path', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH',
                        default='E:/desktop/SIP/data')
    parser.add_argument('--dataset', type=str, default='')#mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['pascal07_six_view'])#'mirflickr_six_view','pascal07_six_view' 30
    # parser.add_argument('--mask-view-ratio', type=list, default=[0.5])
    # # parser.add_argument('--mask-label-ratio', type=list, default=[0.79,0.82,0.85,0.88,0.91,0.94,0.97])
    # parser.add_argument('--mask-label-ratio', type=list, default=[0.5])
    parser.add_argument('--feature_missing_rates', type=list, default=[0.5])
    parser.add_argument('--label_missing_rates', type=float, default=[0.5])
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=3, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--name', type=str, default='final_')
    # Optimization args
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100) #200 for corel5k  100 for iaprtc12 50 for pascal07 30 for espgame
    parser.add_argument('--pre_epochs', type=int, default=0)
    # Training args
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--sigma', type=float, default=0.)
    args = parser.parse_args()
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    lr_list = [1e-3]
    alpha_list = [1e0]#
    beta_list = [1e-3]#1e-3 for corel5k and mirflickr, 1e0 for pascal07, 1e-1 for espgame, 1e0 for iaprtc12
    gamma_list = [0]
    sigma_list = [1e0]#1e0for others ,1e-1 for mirflickr
    if args.lr >= 0.01:
        args.momentumkl = 0.90
    for lr in lr_list:
        args.lr = lr
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma
                    for sigma in sigma_list:
                        args.sigma = sigma
                        for dataset in args.datasets:
                            args.dataset = dataset
                            file_path = osp.join(args.records_dir,args.name+args.dataset+'_VM_' + str(
                                            0.5) + '_LM_' +
                                            str(0.5) + '_T_' +
                                            str(args.training_sample_ratio) + '.txt')
                            args.file_path = file_path
                            existed_params = filterparam(file_path,[-5,-4,-3,-2,-1])
                            if [args.lr,args.alpha,args.beta,args.gamma,args.sigma] in existed_params:
                                print('existed param! beta:{}'.format(args.beta))
                                # continue
                            if __name__ == '__main__':
                                main(args,file_path)