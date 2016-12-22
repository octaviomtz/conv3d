%% Folder with the data
cd('C:\Users\User\UMCG\Imaging FDG\Dec 15 - Datasets for Christmas\scans212_fold1\scansOutNormalized\')

%% Dependencies
addpath(genpath('C:\Users\User\UMCG\Imaging FDG\2016 Feb 7 - First overview from Sanne\Octavio\code\PetICATool'))
addpath('C:\Users\User\UMCG\Matlab mysc\vis3d\')

%% Get the PD and control scans and their sizes
ScansFolder=pwd;

petFiles1=dir([ScansFolder,'\*LDD*.nii']);

nScans1=length(petFiles1);

scansAll=zeros(nScans1,79,95,78);

%% Get all the scan names (to see the order and determine patients and controls for the labels)
scanNames=cell(size(petFiles1,1),1);
for i=1:size(petFiles1,1)
scanNames{i}=petFiles1(i).name;
end

%%
%-- get the PD scans
for i=1:nScans1
    petScanName=petFiles1(i).name;
    petScan=strcat(ScansFolder,'\',petScanName);
    Pfile = spm_vol(petScan);
    Pimg = spm_read_vols(Pfile);
    scansAll(i,:,:,:)=Pimg;
end

%% Visualize
r=ceil(rand*size(petFiles1,1));
vis3d(squeeze(scansAll(r,:,:,:,:)))

%% Get the masks
% We used the mask ontain from the 62 datasets to define 5 ROIs
masksFolder='C:\Users\User\UMCG\Imaging FDG\Dec 14 - Meeting Target\masks';
masks=dir(masksFolder);
mask_1=masks(3).name;
mask2=masks(4).name;
petScan=strcat(masksFolder,'\',mask_1);
Pfile = spm_vol(petScan);
mask1 = spm_read_vols(Pfile);

%% Method to obtain the 5 ROIs
allCenters=zeros(size(mask1));
for i=8:size(allCenters,1)-8
    for j=8:size(allCenters,2)-8
        for k=8:size(allCenters,3)-8
            allCenters(i,j,k)=sum(sum(sum(mask1(i-7:i+8,j-7:j+8,k-7:k+8))));
        end
    end
end
%-- Visualize
vis3d(allCenters)
%-- THE 5 ROIs: The center points of most activations 
cA=[40,26,22];cB=[64,26,52];cC=[60,80,37];cD=[55,52,32];cE=[27,53,32];

%%
c=cA;
scansMini=scansAll(:,c(1)-7:c(1)+8,c(2)-7:c(2)+8,c(3)-7:c(3)+8);
save('fold1scansMini1Out.mat','scansMini','-v7.3')

c=cB;
scansMini=scansAll(:,c(1)-7:c(1)+8,c(2)-7:c(2)+8,c(3)-7:c(3)+8);
save('fold1scansMini2Out.mat','scansMini','-v7.3')

c=cC;
scansMini=scansAll(:,c(1)-7:c(1)+8,c(2)-7:c(2)+8,c(3)-7:c(3)+8);
save('fold1scansMini3Out.mat','scansMini','-v7.3')

c=cD;
scansMini=scansAll(:,c(1)-7:c(1)+8,c(2)-7:c(2)+8,c(3)-7:c(3)+8);
save('fold1scansMini4Out.mat','scansMini','-v7.3')

c=cE;
scansMini=scansAll(:,c(1)-7:c(1)+8,c(2)-7:c(2)+8,c(3)-7:c(3)+8);
save('fold1scansMini5Out.mat','scansMini','-v7.3')


%% For the little scans
%littleScans=scansAll(:,44:59,40:55,33:48);
% littleScans=scansAll(:,44:59,40:55,33:48);
% clearvars -except littleScans
% save('littleScans62Demean.mat','-v7.3')

%% Save
% clearvars -except scansAll
% save('scans62.mat','-v7.3')
