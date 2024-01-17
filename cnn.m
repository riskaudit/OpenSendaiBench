clear, clc, close

%% file paths
gt_path = "/Users/joshuadimasaka/Desktop/PhD/GitHub/" + ...
    "riskaudit/data/groundtruth/METEOR_PROJECT_2002/" + ...
    "AFG_oed_exposure_20200811/tiles/images";
s1_path = "/Users/joshuadimasaka/Desktop/PhD/GitHub/" + ...
    "riskaudit/data/obsvariables/METEOR_PROJECT_2002/" + ...
    "SENTINEL1-DUAL_POL_GRD_HIGH_RES/AFG_oed_exposure_20200811";
s2_path = "/Users/joshuadimasaka/Desktop/PhD/GitHub/" + ...
    "riskaudit/data/obsvariables/METEOR_PROJECT_2002/" + ...
    "SENTINEL-2-MSI_LVL2A/AFG_oed_exposure_20200811";

%% split
rng(1); idx = randperm(100);
idxtrain = idx(1:80)';
idxtest = idx(81:90)';
idxvalid = idx(91:100)';

%% get the global mean and standard deviation for normalization
meanObs = struct();
stdObs = struct();

% s1
s1_listing = struct2table(dir(s1_path));
s1_listing(s1_listing.isdir==0,:) = [];
s1_listing(string(s1_listing.name)=='.',:) = [];
s1_listing(string(s1_listing.name)=='..',:) = [];

% get the mean and std dev of training S1 dataset
s1_vh_all = [];
s1_vv_all = [];
for j = 1:numel(idxtrain)
    pattern = "AFG_"+string(idxtrain(j))+"_of_100";
    for i = 1:100
        if contains(string(s1_listing.name(i)), pattern)
            subfolder = s1_listing.folder(i)+"/"+...
                        s1_listing.name(i);

            fstruct = dir(subfolder+'/2019*_VH.tif');
            [s1_vh,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s1_vh = s1_vh(1:372,1:372);
            s1_vh_all = [s1_vh_all; s1_vh(:)];

            fstruct = dir(subfolder+'/2019*_VV.tif');
            [s1_vv,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s1_vv = s1_vv(1:372,1:372);
            s1_vv_all = [s1_vv_all; s1_vv(:)];
        end 
    end
end
meanObs.s1_vh = mean(s1_vh_all);
stdObs.s1_vh = std(s1_vh_all);
meanObs.s1_vv = mean(s1_vv_all);
stdObs.s1_vv = std(s1_vv_all);
clear -regexp ^s1_v

% s2
s2_listing = struct2table(dir(s2_path));
s2_listing(s2_listing.isdir==0,:) = [];
s2_listing(string(s2_listing.name)=='.',:) = [];
s2_listing(string(s2_listing.name)=='..',:) = [];

% get the mean and std dev of training S2 dataset
s2_b1_all = [];
s2_b2_all = [];
s2_b3_all = [];
s2_b4_all = [];
s2_b5_all = [];
s2_b6_all = [];
s2_b7_all = [];
s2_b8_all = [];
s2_b8a_all = [];
s2_b9_all = [];
s2_b11_all = [];
s2_b12_all = [];
for j = 1:numel(idxtrain)
    pattern = "AFG_"+string(idxtrain(j))+"_of_100";
    for i = 1:100
        if contains(string(s2_listing.name(i)), pattern)
            subfolder = s2_listing.folder(i)+"/"+...
                        s2_listing.name(i);

            fstruct = dir(subfolder+'/2019_B1_*.tif');
            [s2_b1,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b1 = s2_b1(1:372,1:372);
            s2_b1_all = [s2_b1_all; s2_b1(:)];

            fstruct = dir(subfolder+'/2019_B2_*.tif');
            [s2_b2,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b2 = s2_b2(1:372,1:372);
            s2_b2_all = [s2_b2_all; s2_b2(:)];

            fstruct = dir(subfolder+'/2019_B3_*.tif');
            [s2_b3,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b3 = s2_b3(1:372,1:372);
            s2_b3_all = [s2_b3_all; s2_b3(:)];

            fstruct = dir(subfolder+'/2019_B4_*.tif');
            [s2_b4,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b4 = s2_b4(1:372,1:372);
            s2_b4_all = [s2_b4_all; s2_b4(:)];

            fstruct = dir(subfolder+'/2019_B5_*.tif');
            [s2_b5,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b5 = s2_b5(1:372,1:372);
            s2_b5_all = [s2_b5_all; s2_b5(:)];

            fstruct = dir(subfolder+'/2019_B6_*.tif');
            [s2_b6,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b6 = s2_b6(1:372,1:372);
            s2_b6_all = [s2_b6_all; s2_b6(:)];

            fstruct = dir(subfolder+'/2019_B7_*.tif');
            [s2_b7,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b7 = s2_b7(1:372,1:372);
            s2_b7_all = [s2_b7_all; s2_b7(:)];

            fstruct = dir(subfolder+'/2019_B8_*.tif');
            [s2_b8,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b8 = s2_b8(1:372,1:372);
            s2_b8_all = [s2_b8_all; s2_b8(:)];

            fstruct = dir(subfolder+'/2019_B8A_*.tif');
            [s2_b8a,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b8a = s2_b8a(1:372,1:372);
            s2_b8a_all = [s2_b8a_all; s2_b8a(:)];

            fstruct = dir(subfolder+'/2019_B9_*.tif');
            [s2_b9,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b9 = s2_b9(1:372,1:372);
            s2_b9_all = [s2_b9_all; s2_b9(:)];

            fstruct = dir(subfolder+'/2019_B11_*.tif');
            [s2_b11,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b11 = s2_b11(1:372,1:372);
            s2_b11_all = [s2_b11_all; s2_b11(:)];

            fstruct = dir(subfolder+'/2019_B12_*.tif');
            [s2_b12,~] = readgeoraster(subfolder+"/"+fstruct.name);
            s2_b12 = s2_b12(1:372,1:372);
            s2_b12_all = [s2_b12_all; s2_b12(:)];

        end 
    end
end
meanObs.s2_b1 = mean(s2_b1_all);
stdObs.s2_b1 = std(s2_b1_all);
meanObs.s2_b2 = mean(s2_b2_all);
stdObs.s2_b2 = std(s2_b2_all);
meanObs.s2_b3 = mean(s2_b3_all);
stdObs.s2_b3 = std(s2_b3_all);
meanObs.s2_b4 = mean(s2_b4_all);
stdObs.s2_b4 = std(s2_b4_all);
meanObs.s2_b5 = mean(s2_b5_all);
stdObs.s2_b5 = std(s2_b5_all);
meanObs.s2_b6 = mean(s2_b6_all);
stdObs.s2_b6 = std(s2_b6_all);
meanObs.s2_b7 = mean(s2_b7_all);
stdObs.s2_b7 = std(s2_b7_all);
meanObs.s2_b8 = mean(s2_b8_all);
stdObs.s2_b8 = std(s2_b8_all);
meanObs.s2_b8a = mean(s2_b8a_all);
stdObs.s2_b8a = std(s2_b8a_all);
meanObs.s2_b9 = mean(s2_b9_all);
stdObs.s2_b9 = std(s2_b9_all);
meanObs.s2_b11 = mean(s2_b11_all);
stdObs.s2_b11 = std(s2_b11_all);
meanObs.s2_b12 = mean(s2_b12_all);
stdObs.s2_b12 = std(s2_b12_all);
clear -regexp ^s2_b

%% 