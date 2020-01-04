%����scattering wavelet transform �Ĳ���
clc
clear all

filt_opt.J = 3;
filt_opt.L1 = 2;
filt_opt.L2 = 2;
scat_opt.M = 2;
%��ȡͨ��pythonԤ������CT����
load('CT_32_seg');
%��ȡѵ�����ı�ǩ
load('label_train');


%��ѵ��������scattering wavelet transform ����ȡ�Ӵ������� �ֽ�õ�37���Ӵ� ÿ���Ӵ�4������ ��148������
tic;
for i=1:465
    data=reshape(double(train(i,:,:,:))/255,[32,32,32]);
    img=data;
    %����scaterring transform�Ĳ���   
    Wop = my_wavelet_factory_3d(size(img),filt_opt, scat_opt);
    Sx = scat(img, Wop);
    %���б任 �õ�37���Ӵ�
    S = format_scat(Sx);
    for j=1:37
        SS=S(j,:,:,:);
        EE=SS(:).^2;
        %feature1 ��ֵ
        F1(j)=mean(SS(:));
        %feature2 ����
        F2(j)=norm(SS(:),2)^2;
        %feature3 ��׼��
        F3(j)=std(SS(:));
        %feature4 ��
        p=EE./sum(EE);
        F4(j)=-sum(p.*log2(p));
    end
     X_train(i,:)=[F1 F2 F3 F4];
     if(mod(i,20)==0)
         fprintf("ѵ����������ȡ%d/465\n",i)
     end
    end

toc;
disp("ѵ����ͼ��������ȡ���")


    
%����֤������scattering wavelet transform ����ȡ�Ӵ�������  �ֽ�õ�37���Ӵ� ÿ���Ӵ�4������ ��148������
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
         fprintf("���Լ�������ȡ%d/117\n",i)
     end
  end
  toc
  disp("���Լ�������ȡ���")
 
  
  %ѵ������ǩ
Y_train=label;
  %��֪��Ϊɶ ��253�������е��쳣  �����滻�ɵ�252��
    X_train(253,:)=X_train(252,:);
    Y_train(253)=Y_train(252);

  %��ѵ��������֤����148���������й�һ������
for i=1:148
    X_train_norm(:,i)=(X_train(:,i)-mean(X_train(:,i)'))/std(X_train(:,i)');

    X_test_norm(:,i)=(X_test(:,i)-mean(X_train(:,i)'))/std(X_train(:,i)');
end

%��һ��ģ��
m=1



load('model_1.mat');
%%��ѵ�����������
%c1=randperm(numel(1:len));
%a1=c1(1:floor(len*14/15));
%b1=c1(floor(len*11/15)+1:len);
%%��ѵ�������������һ������ΪSVM��ѵ����
% Y_train1=Y_train(a1);
% X_train1 = double(X_train_norm(a1,:));
%%��ѵ�������������һ������ΪSVM�Ĳ��Լ�
% Y_test1=Y_train(b1);
% X_test1 = double(X_train_norm(b1,:));

%����֤����֤��
X_test2=double(X_test_norm);

%ģ�͵�ѵ��  ����ʹ���Ѿ�ѵ���õ�ģ��

%model_SVM1 = svmtrain(Y_train1,train_scat,'-c 2172 -g 0.001    -b 1');
% disp('ѵ����׼ȷ��')
% [predict_label, accuracy, prob] = svmpredict( Y_train1,X_train1 , model_SVM);
% disp('���Լ�׼ȷ��')
% [predict_label, accuracy, prob] = svmpredict(Y_test1,X_test1, model_SVM,'-b 1');

disp('��֤��')
[predict_label, accuracy, prob2] = svmpredict(zeros(117,1),X_test2, model_SVM,'-b 1');


% disp('��һ��ģ�͵Ĳ��Լ�AUC')
%  x=AUC(Y_test1,prob(:,1)')
 
probb(:,m)=prob2(:,1);

%�ڶ���ģ��


m=2
%ѵ������ǩ
Y_train=label;
Y_train(253)=Y_train(252);

load('model_2.mat')
%��ѵ�����������
%c1=randperm(numel(1:len));
%a1=c1(1:floor(len*14/15));
%b1=c1(floor(len*11/15)+1:len);
%��ѵ�������������һ������ΪSVM��ѵ����
% Y_train1=Y_train(a1);
% X_train1 = double(X_train_norm(a1,:));
%��ѵ�������������һ������ΪSVM�Ĳ��Լ�
% Y_test1=Y_train(b1);
% X_test1 = double(X_train_norm(b1,:));

%����֤����֤��
X_test2=double(X_test_norm);

%ģ�͵�ѵ��  ����ʹ���Ѿ�ѵ���õ�ģ��

%model_SVM1 = svmtrain(Y_train1,train_scat,'-c 2172 -g 0.001    -b 1');
% disp('ѵ����׼ȷ��')
% [predict_label, accuracy, prob] = svmpredict( Y_train1,X_train1 , model_SVM2);
% disp('���Լ�׼ȷ��')
% [predict_label, accuracy, prob] = svmpredict(Y_test1,X_test1, model_SVM2,'-b 1');

disp('��֤��')
[predict_label, accuracy, prob2] = svmpredict(zeros(117,1) ,X_test2, model_SVM2,'-b 1');

% disp('��һ��ģ�͵Ĳ��Լ�AUC')
%  x=AUC(Y_test1, prob(:,1)')
 
%ģ�ͼ��ɵõ������ս��
probb(:,m)=prob2(:,1);
prob=mean(probb')';
 
%���csv�ļ�

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