%% file selection
%if possible load default parameters
%build a figure with all default parameters (called menufigure from now on)
[filelist,labellist]=ask_files(); %select all files to work on. (in 2 parts, combined with lable list) 
%update field in menufigure
if isempty(filelist)
    load('fileinfo');
else
    fid=fopen('fil_list','w');
    for i=1:size(filelist,2);
        disp(filelist{1,i});
        fprintf(fid,'%s',filelist{1,i});
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    save('fileinfo.mat','filelist','labellist')
end
%% load files
[DM,w,h,l,V] = load_nifti_images('fil_list',1);%load the data into DM (data matrix)

%% Load GMP
Pmean=spm_select(1,'image','select the image defining the GMP');
Vmean=spm_vol(Pmean);
GMP_vol=spm_read_vols(Vmean);
GMP=GMP_vol(:);

%% Load Mask
Pmask=spm_select(1,'image','select mask image');
Vmask=spm_vol(Pmask);
Mask_vol=spm_read_vols(Vmask);
ind = find(Mask_vol);
%%checks and balances for incompatible dimensions? 

%%
allDM=DM;

%% Change shape
DM_=zeros(size(DM));
for i=1:size(allDM,2)
    DM_(:,i)=reshape(DM(:,i),prod(V(1).dim),size(DM(:,i),4));
    DM_temp=DM_(:,i);
    DM_temp=DM_temp(ind,:);
    DM_shape(:,i)=DM_temp;
end

%% NEW LOG PART
DM_shape=log(DM_shape);
imagDM1=zeros(V(1).dim);
for k=1:size(DM_shape,2);
	imagDM1(ind)=DM_shape(:,k);
    [~,f,e]=fileparts(filelist{k});
	fnameout=['ocL_' f e];
	write_nii_file(imagDM1, 'file', fnameout, 'mat',V.mat); 
end

%% Demean within subject
DM_LD=zeros(size(DM_shape));
for k=1:size(DM_LD,2)
    DM_LD(:,k)=detrend(DM_shape(:,k),'constant');
end
    
% Save LD files 
imagDM1=zeros(V(1).dim);
for k=1:size(DM_LD,2);
	imagDM1(ind)=DM_LD(:,k);
    [~,f,e]=fileparts(filelist{k});
	fnameout=['oc1LD_' f e];
	write_nii_file(imagDM1, 'file', fnameout, 'mat',V.mat); 
end

%% apply demean per voxel, using mean determined while building the pattern
GMPM=GMP(ind,:);

imagDM1=zeros(V(1).dim);
DM_LDD=zeros(size(DM_LD));
for k=1:size(DM_LDD,2)
    DM_LDD(:,k)=DM_LD(:,k)-repmat(GMPM,[1 size(DM_LD(:,k),2)]);
    imagDM1(ind)=DM_LDD(:,k);
    [~,f,e]=fileparts(filelist{k});
	fnameout=['oc-LDD_' f e];
	write_nii_file(imagDM1, 'file', fnameout, 'mat',V.mat); 
end


%% STOP HERE