clc;
clear;
%% ��������
data=importdata('mfeat-kar');
for i=0:9
    for j=1:200
    data(i*200+j:((i+1)*200),65)=i; 
    end
end
%% �������ѵ�����Ͳ��Լ�
N=size(data);
ind = randperm(N(1)); %����һ�����������
sample_train=data(ind(1:N(1)*0.5),1:65); %�����ѡ��ѵ������, 50%
sample_test=data(ind(N(1)*0.5+1:end),1:65); %�����ѡ����������, 50%

%% ��ѵ��������Ϊ�б�ǵ����ݼ�L���ޱ�ǵ����ݼ�U   L/U=90%
    sample_train_L=sample_train(1:100,:);       %�õ��б��ѵ������ռѵ����10%
    sample_train_U=sample_train(101:end,1:64);  %�õ��ޱ��ѵ������ռѵ����90%
    L1=zeros(1000,33);
    L2=zeros(1000,33);
    %��L��Ϊ������ͼL1 �� L2
    L1(1:100,1:32)=sample_train_L(1:100,1:32);
    L1(1:100,33)=sample_train_L(1:100,65);
    L2(1:100,1:32)=sample_train_L(1:100,33:64);
    L2(1:100,33)=sample_train_L(1:100,65);
     for i=1:9
        U11=sample_train_U(((i-1)*100+1):(i*100),1:32);                 % �õ�һ����ʼ���ޱ���Ӽ�U'
        meas1=L1(1:(i*100),1:32);
        species1=L1(1:(i*100),33);
        by1=fitcnb(meas1,species1);
        R1=by1.predict(U11);
        result1(1:100,1:32)=L1(1:100,1:32);
        result1(1:100,33)=R1(1:100,1);

        U12=sample_train_U(((i-1)*100+1):(i*100),33:64);            
        meas2=L2(1:(i*100),1:32);
        species2=L2(1:(i*100),33);
        by2=fitcnb(meas2,species2);
        R2=by2.predict(U12);
        result2(1:100,1:32)=L2(1:100,1:32);
        result2(1:100,33)=R2(1:100,1);

        L1((i*100+1):((i+1)*100),:)=result2(1:100,:);
        L2((i*100+1):((i+1)*100),:)=result1(1:100,:);
     end

%% h1�Բ��Լ����з���
sample_test_t1=sample_test(:,1:32);   %���Լ�����������
sample_test_t2=sample_test(:,33:64); 
sample_test_c=sample_test(:,65);     %���Լ������

last_r1=by1.predict(sample_test_t1);
cMat1=confusionmat(sample_test_c,last_r1);
r=diag(cMat1);
accuracy1=(sum(r))/1000;
%% h2�Բ��Լ����з���
last_r2=by2.predict(sample_test_t2);
cMat2=confusionmat(sample_test_c,last_r2);
r2=diag(cMat2);
accuracy2=(sum(r2))/2000;
disp('accuracy=  ');
if(accuracy1>=accuracy2)
    disp(accuracy1);
else
    disp(accuracy2);
end