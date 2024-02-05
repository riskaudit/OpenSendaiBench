import os
import re
import glob
import cv2
import rasterio
import scipy
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from constants import labels

class OpenSendaiBenchDataset(Dataset):
    """
    An implementation of a PyTorch dataset for loading pairs of observable variables and ground truth labels.
    Inspired by https://pytorch.org/tutorials/beginner/data_loading_tutorial.html.
    """
    def __init__(self, 
                 obsvariables_path: str, groundtruth_path: str, 
                 ifile: list, country: str, signal: list, lognorm_dist: dict, transform: transforms = None):
        """
        Constructs an OpenSendaiBenchDataset.
        :param obsvariables_path: Path to the source folder of observable variables
        :param groundtruth_path: Path to the source folder of corresponding ground truth labels
        :param transform: Callable transformation to apply to images upon loading
        """
        self.obsvariables_path = obsvariables_path
        self.groundtruth_path = groundtruth_path
        self.ifile = ifile
        self.country = country
        self.signal = signal
        self.transform = transform
        self.lognorm_dist = lognorm_dist

    def __len__(self):
        """
        Implements the len(SeaIceDataset) magic method. Required to implement by Dataset superclass.
        When training/testing, this method tells our training loop how much longer we have to go in our Dataset.
        :return: Length of OpenSendaiBenchDataset
        """
        return len(self.ifile)

    def __getitem__(self, i: int):
        """
        Implements the OpenSendaiBenchDataset[i] magic method. Required to implement by Dataset superclass.
        When training/testing, this method is used to actually fetch data.
        :param i: Index of which image pair to fetch
        :return: Dictionary with pairs of observable variables and ground truth labels.
        """
        k = self.ifile[i]
        obsvariable = np.zeros([len(self.signal),368,368])
        for s in range(len(self.signal)):
            for file in glob.glob(str(self.obsvariables_path+
                                    '**/'+self.country+'_*/'+self.country+'_'+
                                    str(k)+'_'+'of_*/2019*_'+self.signal[s]+'.tif')):
                a = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                a = cv2.resize(a, (368,368), interpolation = cv2.INTER_NEAREST)
                obsvariable[s,:,:] = np.nan_to_num(a.reshape(1,a.shape[0],a.shape[1]))
                
        groundtruth = np.zeros([len(labels[self.country]),8,8])
        for w in range(len(labels[self.country])): 
            for file in glob.glob(str(self.groundtruth_path+
                                      self.country+'*/tiles/images/'+
                                      self.country+'_nbldg_'+labels[self.country][w]+'_'+str(k)+'_'+'of_'+'*.tif')):
                a = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                groundtruth[w,:,:] = np.nan_to_num(self.lognorm_dist[labels[self.country][w]]['modelfit'].cdf(a.reshape(1,a.shape[0],a.shape[1])))

        obsvariable = torch.from_numpy(obsvariable).float() #.unsqueeze(0)
        # obsvariable_8x8 = torch.from_numpy(obsvariable_8x8).float()
        groundtruth = torch.from_numpy(groundtruth).float() #.unsqueeze(0)
    
        sample = {"obsvariable": obsvariable,
                  "groundtruth": groundtruth}
        if self.transform:
            sample = {"obsvariable": self.transform(obsvariable),
                      "groundtruth": self.transform(groundtruth).squeeze(0).long()}
        return sample

    def visualise(self, i):
        """
        Allows us to visualise a particular SAR/chart pair.
        :param i: Index of which image pair to visualise
        :return: None
        """
        k = self.ifile[i]
        sample = self[k]
        fig1, axs1 = plt.subplots(1,len(self.signal))
        for s in range(len(self.signal)):
            axs1[s].imshow(sample['obsvariable'][s,:,:])
            axs1[s].set_title(str(self.signal[s]))
            axs1[s].set_xticks([])
            axs1[s].set_yticks([])
        plt.tight_layout()
 
        fig2, axs2 = plt.subplots(1,len(labels[self.country]))
        for w in range(len(labels[self.country])): 
            axs2[w].imshow(sample['groundtruth'][w,:,:])
            axs2[w].set_title(labels[self.country][w])
            axs2[w].set_xticks([])
            axs2[w].set_yticks([])
        plt.tight_layout()

class OpenSendaiBenchDatasetGlobal(Dataset):
    """
    An implementation of a PyTorch dataset for loading pairs of observable variables and ground truth labels.
    Inspired by https://pytorch.org/tutorials/beginner/data_loading_tutorial.html.
    """
    def __init__(self, 
                 obsvariables_path: str, 
                 FilePathList: list, 
                 bldgtype: str,
                 signal: list,
                 lognorm_dist_list: dict, 
                 transform: transforms = None):
        """
        Constructs an OpenSendaiBenchDataset.
        :param obsvariables_path: Path to the source folder of observable variables
        :param groundtruth_path: Path to the source folder of corresponding ground truth labels
        :param transform: Callable transformation to apply to images upon loading
        """
        self.obsvariables_path = obsvariables_path
        self.FilePathList = FilePathList
        self.bldgtype = bldgtype
        self.lognorm_dist_list = lognorm_dist_list
        self.transform = transform
        self.signal = signal #['VH', 'VV', 'aerosol', 'blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'vapor', 'swir1', 'swir2']

    def __len__(self):
        """
        Implements the len(SeaIceDataset) magic method. Required to implement by Dataset superclass.
        When training/testing, this method tells our training loop how much longer we have to go in our Dataset.
        :return: Length of OpenSendaiBenchDataset
        """
        return len(self.FilePathList)

    def __getitem__(self, i: int):
        """
        Implements the OpenSendaiBenchDataset[i] magic method. Required to implement by Dataset superclass.
        When training/testing, this method is used to actually fetch data.
        :param i: Index of which image pair to fetch
        :return: Dictionary with pairs of observable variables and ground truth labels.
        """
        file1 = self.FilePathList[i]
        
        start = file1.find('2002/') + 5
        end = file1.find('_oed_exposure', start)
        country = file1[start:end]

        start = file1.find(str('_'+self.bldgtype+'_')) + 2 + len(self.bldgtype)
        end = file1.find('_of_', start)
        k = file1[start:end]

        obsvariable = np.zeros([3,368,368]) # to modify later
        for s in range(len(self.signal)):
            for file2 in glob.glob(str(self.obsvariables_path+
                                    '**/'+country+'_*/'+country+'_'+
                                    str(k)+'_'+'of_*/2019*_'+self.signal[s]+'.tif')):
                if self.signal[s] == 'RGB':
                    a = rasterio.open(file2).read()
                    for x in range(3):
                        b = cv2.resize(a[x,:,:], (368,368), interpolation = cv2.INTER_NEAREST)
                        obsvariable[x,:,:] = np.nan_to_num(b.reshape(1,b.shape[0],b.shape[1]))
                else:
                    a = cv2.imread(file2, cv2.IMREAD_UNCHANGED)
                    a = cv2.resize(a, (368,368), interpolation = cv2.INTER_NEAREST)
                    obsvariable[s,:,:] = np.nan_to_num(a.reshape(1,a.shape[0],a.shape[1]))


        a = cv2.imread(file1, cv2.IMREAD_UNCHANGED)
        groundtruth = np.nan_to_num(self.lognorm_dist_list[country][self.bldgtype]['modelfit'].cdf(a.reshape(1,a.shape[0],a.shape[1])))

        obsvariable = torch.from_numpy(obsvariable).float() 
        groundtruth = torch.from_numpy(groundtruth).float() 
    
        sample = {"obsvariable": obsvariable, 
                  "groundtruth": groundtruth,
                  "mu": self.lognorm_dist_list[country][self.bldgtype]['mu'],
                  "sigma": self.lognorm_dist_list[country][self.bldgtype]['sigma']}

        return sample

def fitlognorm(groundtruth_path: str):
    lognorm_dist_list = {}
    if not os.path.exists('./lognorm/'): 
        os.makedirs('./lognorm/')
    for icountry in range(len(list(labels.keys()))):
        country = list(labels.keys())[icountry]
        datapath = groundtruth_path+str(country)+'_oed_exposure_20200811/'
        # f, ax = plt.subplots(ncols=1, nrows=len(labels[country]), figsize=(7, 5*len(labels[country])))
        
        lognorm_dist_list[country] = {}
        # get the lognormal fitting parameters for each building type
        for ibldgtype in range(len(labels[country])):
            nbldg_file = datapath+'attr_rasterized/'+str(country)+'_nbldg_'+str(labels[country][ibldgtype])+'.tif'
            mask_file = datapath+str(country)+"_country.tif"
            nb = cv2.imread(nbldg_file, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
            nb_masked = nb.flatten()[mask.flatten()>0]
            nb_masked = nb_masked[nb_masked != 0]

            x_exp = nb_masked
            mu = np.mean(np.log(nb_masked))
            sigma = np.std(np.log(nb_masked)) 
            mu_exp = np.exp(mu) 
            sigma_exp = np.exp(sigma)
            fitting_params_lognormal = scipy.stats.lognorm.fit(x_exp, floc=0, scale=mu_exp)
            lognorm_dist_fitted = scipy.stats.lognorm(*fitting_params_lognormal)
            t = np.linspace(np.min(x_exp), np.max(x_exp), 100)
            lognorm_dist = scipy.stats.lognorm(s=sigma, loc=0, scale=np.exp(mu))
            lognormal_test = scipy.stats.kstest(x_exp, lognorm_dist.cdf)

            lognorm_dist_list[country][labels[country][ibldgtype]] = {}
            lognorm_dist_list[country][labels[country][ibldgtype]]['modelfit'] = lognorm_dist
            lognorm_dist_list[country][labels[country][ibldgtype]]['mu'] = mu
            lognorm_dist_list[country][labels[country][ibldgtype]]['sigma'] = sigma
            lognorm_dist_list[country][labels[country][ibldgtype]]['KStest_stat'] = lognormal_test.statistic
            lognorm_dist_list[country][labels[country][ibldgtype]]['KStest_pvalue'] = lognormal_test.pvalue

        #     sns.distplot(x_exp, ax=ax[ibldgtype], norm_hist=True, kde=False,
        #                 label='Data')
        #     ax[ibldgtype].plot(t, lognorm_dist_fitted.pdf(t), lw=2, color='r',
        #             label='Fitted Model X~LogNorm(mu={0:.1f}, sigma={1:.1f})'.format(lognorm_dist_fitted.mean(), lognorm_dist_fitted.std()))
        #     ax[ibldgtype].plot(t, lognorm_dist.pdf(t), lw=2, color='g', ls=':',
        #             label='Original Model X~LogNorm(mu={0:.1f}, sigma={1:.1f})'.format(lognorm_dist.mean(), lognorm_dist.std()))
        #     ax[ibldgtype].title.set_text(str(labels[country][ibldgtype]))
        #     ax[ibldgtype].legend(loc='upper right')

        # f.savefig('./lognorm/'+str(country)+'.png')

    return lognorm_dist_list