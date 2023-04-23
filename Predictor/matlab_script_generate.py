import argparse

parser = argparse.ArgumentParser(description='Matlab Scripts Generation')
parser.add_argument('--sample_folder', default='./logs', type=str, help='path to the folder includes all sample files')
parser.add_argument('--save_folder', default='./logs', type=str, help='path to folder for saving .mat from Matlab')

args = parser.parse_args()

with open('matlab_train.m', 'w+') as train_file:
    train_file.write('''
clc;
clear;

folder=\'''' + args.sample_folder + '''/\';
csvlist=dir([folder,'*_final.csv']);
savefolder=\'''' + args.save_folder + '''/\';
mkdir(savefolder);
csvnum = length(csvlist);


for i = 1:csvnum
    file_name=[folder,csvlist(i).name];
    M{i}=csvread(file_name);
end


for i = 1:csvnum
    cur_M=M{i};
    cur_M=transpose(cur_M);
    cur_P=cur_M(1:2,:);
    cur_T=cur_M(3,:);
    cur_name=[savefolder, csvlist(i).name(1:end-4),'.mat'];

    cur_T=power(cur_T,-(1/8));

    [p1,minp,maxp,t1,mint,maxt]=premnmx(cur_P,cur_T);

    net=newff(minmax(cur_P),[2,4,1],{'tansig','tansig','purelin'},'trainlm');

    net.trainParam.epochs = 5000;

    net.trainParam.goal=0.0000001;

    [net,tr]=train(net,p1,t1);
    save(cur_name,'net','mint','maxt','minp','maxp');
    
end

    ''')

with open('matlab_train_every.m', 'w+') as train_file:
    train_file.write('''
clc;
clear;

folder=\'''' + args.sample_folder + '''/\';
name='mobilenet_v1_result_cpu_28_3_3_2_2_1_1_False_g_final'; % need to be changed to the specific mat for re-training
file_name=[folder,name,'.csv'];
savefolder=\'''' + args.save_folder + '''/\';

cur_M = csvread(file_name);

cur_M=transpose(cur_M);
cur_P=cur_M(1:2,:);
cur_T=cur_M(3,:);
cur_name=[savefolder, name,'.mat'];

cur_T=power(cur_T,-(1/8));

[p1,minp,maxp,t1,mint,maxt]=premnmx(cur_P,cur_T);

net=newff(minmax(cur_P),[2,4,1],{'tansig','tansig','purelin'},'trainlm');

net.trainParam.epochs = 5000;

net.trainParam.goal=0.0000001;

[net,tr]=train(net,p1,t1);
save(cur_name,'net','mint','maxt','minp','maxp');

    ''')