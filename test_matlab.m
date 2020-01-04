%设置scattering wavelet transform 的参数
clc
clear all

filt_opt.J = 3;
filt_opt.L1 = 2;
filt_opt.L2 = 2;
scat_opt.M = 2;
%读取通过python预处理后的CT序列
load('CT_32_seg');
%读取训练集的标签
load('label_train');


%对训练集进行scattering wavelet transform 并提取子带的特征 分解得到37个子带 每个子带4个特征 共148个特征
tic;
for i=1:465
    data=reshape(double(train(i,:,:,:))/255,[32,32,32]);
    img=data;
    %设置scaterring transform的参数   
    Wop = my_wavelet_factory_3d(size(img),filt_opt, scat_opt);
    Sx = scat(img, Wop);
    %进行变换 得到37个子带
    S = format_scat(Sx);
    for j=1:37
        SS=S(j,:,:,:);
        EE=SS(:).^2;
        %feature1 均值
        F1(j)=mean(SS(:));
        %feature2 能量
        F2(j)=norm(SS(:),2)^2;
        %feature3 标准差
        F3(j)=std(SS(:));
        %feature4 熵
        p=EE./sum(EE);
        F4(j)=-sum(p.*log2(p));
    end
     X_train(i,:)=[F1 F2 F3 F4];
     if(mod(i,20)==0)
         fprintf("训练集特征提取%d/465\n",i)
     end
    end

toc;
disp("训练集图像特征提取完毕")


    
%对验证集进行scattering wavelet transform 并提取子带的特征  分解得到37个子带 每个子带4个特征 共148个特征
tic
for i=1:117
    data=reshape(double(test(i,:,:,:))/255,[32,32,32]);
    img=data;    
    Wop = my_wavelet_factory_3d(size(img),filt_opt, scat_opt);
    Sx = scat(img, Wop);
    S = format_scat(Sx);
    for j=1:37
        SS=S(j,:,:,:);
        EE=SS(:).^2;
        F1(j)=mean(SS(:));
        F2(j)=norm(SS(:),2)^2;
        F3(j)=std(SS(:));
        p=EE./sum(EE);
        F4(j)=-sum(p.*log2(p));
    end
        X_test(i,:)=[F1 F2 F3 F4];
     if(mod(i,20)==0)
         fprintf("测试集特征提取%d/117\n",i)
     end
  end
  toc
  disp("测试集特征提取完毕")
 
  
  %训练集标签
Y_train=label;
  %不知道为啥 第253个数据有点异常  把它替换成第252个
    X_train(253,:)=X_train(252,:);
    Y_train(253)=Y_train(252);

  %对训练集和验证集的148个特征进行归一化处理
for i=1:148
    X_train_norm(:,i)=(X_train(:,i)-mean(X_train(:,i)'))/std(X_train(:,i)');

    X_test_norm(:,i)=(X_test(:,i)-mean(X_train(:,i)'))/std(X_train(:,i)');
end

%第一个模型
m=1



load('model_1.mat');
%%将训练集随机打乱
%c1=randperm(numel(1:len));
%a1=c1(1:floor(len*14/15));
%b1=c1(floor(len*11/15)+1:len);
%%从训练集中随机划分一部分作为SVM的训练集
% Y_train1=Y_train(a1);
% X_train1 = double(X_train_norm(a1,:));
%%从训练集中随机划分一部分作为SVM的测试集
% Y_test1=Y_train(b1);
% X_test1 = double(X_train_norm(b1,:));

%待验证的验证集
X_test2=double(X_test_norm);

%模型的训练  这里使用已经训练好的模型

%model_SVM1 = svmtrain(Y_train1,train_scat,'-c 2172 -g 0.001    -b 1');
% disp('训练集准确率')
% [predict_label, accuracy, prob] = svmpredict( Y_train1,X_train1 , model_SVM);
% disp('测试集准确率')
% [predict_label, accuracy, prob] = svmpredict(Y_test1,X_test1, model_SVM,'-b 1');

disp('验证集')
[predict_label, accuracy, prob2] = svmpredict(zeros(117,1),X_test2, model_SVM,'-b 1');


% disp('第一个模型的测试集AUC')
%  x=AUC(Y_test1,prob(:,1)')
 
probb(:,m)=prob2(:,1);

%第二个模型


m=2
%训练集标签
Y_train=label;
Y_train(253)=Y_train(252);

load('model_2.mat')
%将训练集随机打乱
%c1=randperm(numel(1:len));
%a1=c1(1:floor(len*14/15));
%b1=c1(floor(len*11/15)+1:len);
%从训练集中随机划分一部分作为SVM的训练集
% Y_train1=Y_train(a1);
% X_train1 = double(X_train_norm(a1,:));
%从训练集中随机划分一部分作为SVM的测试集
% Y_test1=Y_train(b1);
% X_test1 = double(X_train_norm(b1,:));

%待验证的验证集
X_test2=double(X_test_norm);

%模型的训练  这里使用已经训练好的模型

%model_SVM1 = svmtrain(Y_train1,train_scat,'-c 2172 -g 0.001    -b 1');
% disp('训练集准确率')
% [predict_label, accuracy, prob] = svmpredict( Y_train1,X_train1 , model_SVM2);
% disp('测试集准确率')
% [predict_label, accuracy, prob] = svmpredict(Y_test1,X_test1, model_SVM2,'-b 1');

disp('验证集')
[predict_label, accuracy, prob2] = svmpredict(zeros(117,1) ,X_test2, model_SVM2,'-b 1');

% disp('第一个模型的测试集AUC')
%  x=AUC(Y_test1, prob(:,1)')
 
%模型集成得到的最终结果
probb(:,m)=prob2(:,1);
prob=mean(probb')';
 
%输出csv文件

name=["candidate11"
"candidate13"
"candidate15"
"candidate17"
"candidate22"
"candidate26"
"candidate33"
"candidate40"
"candidate42"
"candidate49"
"candidate56"
"candidate59"
"candidate60"
"candidate68"
"candidate75"
"candidate76"
"candidate77"
"candidate79"
"candidate85"
"candidate99"
"candidate100"
"candidate103"
"candidate107"
"candidate108"
"candidate112"
"candidate116"
"candidate117"
"candidate118"
"candidate128"
"candidate129"
"candidate132"
"candidate137"
"candidate144"
"candidate146"
"candidate149"
"candidate165"
"candidate167"
"candidate170"
"candidate174"
"candidate178"
"candidate189"
"candidate191"
"candidate194"
"candidate195"
"candidate197"
"candidate201"
"candidate204"
"candidate223"
"candidate239"
"candidate251"
"candidate252"
"candidate259"
"candidate264"
"candidate268"
"candidate274"
"candidate275"
"candidate278"
"candidate284"
"candidate286"
"candidate290"
"candidate300"
"candidate301"
"candidate302"
"candidate309"
"candidate312"
"candidate329"
"candidate333"
"candidate344"
"candidate346"
"candidate348"
"candidate354"
"candidate357"
"candidate360"
"candidate363"
"candidate368"
"candidate373"
"candidate384"
"candidate390"
"candidate394"
"candidate396"
"candidate397"
"candidate398"
"candidate402"
"candidate403"
"candidate412"
"candidate424"
"candidate427"
"candidate431"
"candidate439"
"candidate440"
"candidate450"
"candidate451"
"candidate455"
"candidate456"
"candidate458"
"candidate459"
"candidate463"
"candidate466"
"candidate483"
"candidate486"
"candidate489"
"candidate491"
"candidate493"
"candidate498"
"candidate499"
"candidate502"
"candidate509"
"candidate513"
"candidate524"
"candidate530"
"candidate544"
"candidate556"
"candidate563"
"candidate564"
"candidate565"
"candidate580"
"candidate582"
];
columns = {'Id','Predicted'};
data = table(name,prob, 'VariableNames', columns)
writetable(data, 'Submission.csv')