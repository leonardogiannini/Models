import numpy
from random import randint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import LogNorm

debug=True
remove=False

def mean_and_std (tvars):
    tcol_mean = numpy.nanmean(tvars,axis=0)            
    tcol_std = numpy.nanstd(tvars,axis=0)
    return tcol_mean, tcol_std

def make_vars(f):
    
    zipfile=numpy.load(f)
    
    jvars=zipfile['arr_0']
    tvars=zipfile['arr_2']
    
    if debug:
        print jvars.shape, "shape of javrs array"
        print tvars.shape, "shape of tavrs array"
        
    tinds = numpy.where(numpy.isnan(tvars))
    tevents_to_remove=numpy.unique(tinds[0])
    
    jinds = numpy.where(numpy.isnan(jvars))
    jevents_to_remove=numpy.unique(jinds[0])
    
    events_to_remove=numpy.unique(numpy.concatenate((tevents_to_remove,jevents_to_remove)))
    events_to_remove=numpy.sort(events_to_remove)
    events_to_remove=events_to_remove[::-1]
    
    if remove:
        for ev in events_to_remove:            
            tvars=numpy.delete(tvars, ev,0)
            jvars=numpy.delete(jvars, ev,0)
            if debug: print "removed event:  ", ev
    else:    
        tvars[tinds]=-10

    tvars=numpy.swapaxes(tvars, 1,2)

    svars=tvars[:,:,0:21] #solo seed_vars
    tvars=tvars[:,:,21:741]#solo quelle utili ovvero nearTracks_
    
    tvars=tvars.reshape((len(tvars),10,20,36))
    
    s_rawpt=svars[:,:,0]
    t_rawpt=tvars[:,:,:,0]

    if debug: print tvars.shape, svars.shape, jvars.shape, s_rawpt.shape, t_rawpt.shape
        
    return tvars, svars, jvars, s_rawpt, t_rawpt


#HERE a sample file is taken as the refernce for mean and std values (for files of the sample)
sample_file="/gpfs/ddn/users/lgiannini/TTbar_samples/test_file.npz"
sample_tvars, sample_svars, sample_jvars , s_rawpt, t_rawpt= make_vars(sample_file)

tcol_mean, tcol_std = mean_and_std(sample_tvars)
scol_mean, scol_std = mean_and_std(sample_svars)

#first track in jet gives the normalization
tcol_mean2=tcol_mean[0,0,:]
tcol_std2=tcol_std[0,0,:]

scol_mean2=scol_mean[0,:]
scol_std2=scol_std[0,:]

#track probabilities unnormalized (they are already)
scol_mean2[17]=0
scol_mean2[18]=0
scol_std2[17]=1
scol_std2[18]=1

def make_varsNormalized(f):
    
    zipfile=numpy.load(f)
    
    jvars=zipfile['arr_0']
    tvars=zipfile['arr_2']
    
    if debug:
        print jvars.shape, "shape of javrs array"
        print tvars.shape, "shape of tavrs array"
        
    tinds = numpy.where(numpy.isnan(tvars))
    tevents_to_remove=numpy.unique(tinds[0])
    
    jinds = numpy.where(numpy.isnan(jvars))
    jevents_to_remove=numpy.unique(jinds[0])
    
    events_to_remove=numpy.unique(numpy.concatenate((tevents_to_remove,jevents_to_remove)))
    events_to_remove=numpy.sort(events_to_remove)
    events_to_remove=events_to_remove[::-1]
    
    if remove:
        for ev in events_to_remove:            
            tvars=numpy.delete(tvars, ev,0)
            jvars=numpy.delete(jvars, ev,0)
            if debug: print "removed event:  ", ev
    else:    
        tvars[tinds]=-10

    if debug: print "mean and std corrections"
    
    tvars=numpy.swapaxes(tvars, 1,2)

    svars=tvars[:,:,0:21] #solo seed_vars
    tvars=tvars[:,:,21:741]#solo quelle utili ovvero nearTracks_
    
    tvars=tvars.reshape((len(tvars),10,20,36))
    
    s_rawpt=svars[:,:,0]
    t_rawpt=tvars[:,:,:,0]

    if debug: print tvars.shape, svars.shape, jvars.shape, s_rawpt.shape, t_rawpt.shape
    
    #normalization is done here (show plots to debug)
    
    tvars=tvars-(tvars!=0)*tcol_mean2
    tvars=tvars*(tvars!=0)/tcol_std2
    
    svars=svars-(svars!=0)*scol_mean2
    svars=svars*(svars!=0)/scol_std2
        
    return tvars, svars, jvars, s_rawpt, t_rawpt

#############################################################################################################################

def make_varsTransformed(f):
    
    zipfile=numpy.load(f)
    
    jvars=zipfile['arr_0']
    tvars=zipfile['arr_2']
    
    if debug:
        print jvars.shape, "shape of javrs array"
        print tvars.shape, "shape of tavrs array"
        
    tinds = numpy.where(numpy.isnan(tvars))
    tevents_to_remove=numpy.unique(tinds[0])
    
    jinds = numpy.where(numpy.isnan(jvars))
    jevents_to_remove=numpy.unique(jinds[0])
    
    events_to_remove=numpy.unique(numpy.concatenate((tevents_to_remove,jevents_to_remove)))
    events_to_remove=numpy.sort(events_to_remove)
    events_to_remove=events_to_remove[::-1]
    
    if remove:
        for ev in events_to_remove:            
            tvars=numpy.delete(tvars, ev,0)
            jvars=numpy.delete(jvars, ev,0)
            if debug: print "removed event:  ", ev
    else:    
        tvars[tinds]=-10

    tvars=numpy.swapaxes(tvars, 1,2)

    svars=tvars[:,:,0:21] #solo seed_vars
    tvars=tvars[:,:,21:741]#solo quelle utili ovvero nearTracks_
    
    tvars=tvars.reshape((len(tvars),10,20,36))
    
    s_rawpt=svars[:,:,0]
    t_rawpt=tvars[:,:,:,0]
    
    #after raw_pt is assigned--> do this

    if debug: print svars[:,:,0]
    svars[:,:,0]=(svars[:,:,0]!=0)/svars[:,:,0]#"seed_pt"  normalized - logscale    
    if debug: print svars[:,:,0]
    #"seed_eta" normalized
    #"seed_phi" normalized
    #"seed_mass" normalized
    svars[:,:,4]=(numpy.log(abs(svars[:,:,4]))*(svars[:,:,4]!=0)+numpy.min(abs(svars[:,:,4])))*numpy.sign(svars[:,:,4])#"seed_dz" normalized - symmetric log
    svars[:,:,5]=(numpy.log(abs(svars[:,:,5]))*(svars[:,:,5]!=0)+numpy.min(abs(svars[:,:,5])))*numpy.sign(svars[:,:,5])#"seed_dxy" normalized - symmetric log
    svars[:,:,6]=(svars[:,:,6]!=0)*numpy.log(svars[:,:,6])#"seed_3Dip" normalized - log
    svars[:,:,7]=(svars[:,:,7]!=0)*numpy.log(svars[:,:,7])#"seed_3Dsip" normalized - log
    svars[:,:,8]=(svars[:,:,8]!=0)*numpy.log(svars[:,:,8])#"seed_2Dip" normalized - log
    svars[:,:,9]=(svars[:,:,9]!=0)*numpy.log(svars[:,:,9])#"seed_2Dsip" normalized - log
    svars[:,:,10]=(numpy.log(abs(svars[:,:,10]))*(svars[:,:,10]!=0)+numpy.min(abs(svars[:,:,10])))*numpy.sign(svars[:,:,10])#"seed_3DsignedIp" normalized - symmetric log
    svars[:,:,11]=(numpy.log(abs(svars[:,:,11]))*(svars[:,:,11]!=0)+numpy.min(abs(svars[:,:,11])))*numpy.sign(svars[:,:,11])#"seed_3DsignedSip" normalized - symmetric log
    svars[:,:,12]=(numpy.log(abs(svars[:,:,12]))*(svars[:,:,12]!=0)+numpy.min(abs(svars[:,:,12])))*numpy.sign(svars[:,:,12])#"seed_2DsignedIp" normalized - symmetric log
    svars[:,:,13]=(numpy.log(abs(svars[:,:,13]))*(svars[:,:,13]!=0)+numpy.min(abs(svars[:,:,13])))*numpy.sign(svars[:,:,13])#"seed_2DsignedSip" normalized - symmetric log
    #"seed_3D_TrackProbability" -  normalized
    #"seed_2D_TrackProbability" -  normalized
    svars[:,:,16]=(svars[:,:,16]!=0)*numpy.log(svars[:,:,16])#"seed_chi2reduced"  normalized - log
    #"seed_nPixelHits" 
    #"seed_nHits"
    svars[:,:,19]=(svars[:,:,19]!=0)*numpy.log(svars[:,:,19])#"seed_jetAxisDistance" normalized - log
    svars[:,:,20]=(svars[:,:,20]!=0)*numpy.log(svars[:,:,20])#"seed_jetAxisDlength" normalized - log
    
    #some nan appear due to inversion and log
    svars=numpy.nan_to_num(svars)  
    
    tvars[:,:,:,0]=(tvars[:,:,:,0]!=0)/tvars[:,:,:,0]#"nearTracks_pt" normalized - logscale
    #"nearTracks_eta" normalized
    #"nearTracks_phi" normalized
    tvars[:,:,:,3]=(numpy.log(abs(tvars[:,:,:,3]))*(tvars[:,:,:,3]!=0)+numpy.min(abs(tvars[:,:,:,3])))*numpy.sign(tvars[:,:,:,3])#"nearTracks_dz" normalized - symmetric log
    tvars[:,:,:,4]=(numpy.log(abs(tvars[:,:,:,4]))*(tvars[:,:,:,4]!=0)+numpy.min(abs(tvars[:,:,:,4])))*numpy.sign(tvars[:,:,:,4])#"nearTracks_dxy" normalized - symmetric log
    #"nearTracks_mass" normalized
    tvars[:,:,:,6]=(tvars[:,:,:,6]!=0)*numpy.log(tvars[:,:,:,6])#"nearTracks_3D_ip" normalized - logscale
    tvars[:,:,:,7]=(tvars[:,:,:,7]!=0)*numpy.log(tvars[:,:,:,7])#"nearTracks_3D_sip", normalized - logscale
    tvars[:,:,:,8]=(tvars[:,:,:,8]!=0)*numpy.log(tvars[:,:,:,8])#"nearTracks_2D_ip" normalized - logscale
    tvars[:,:,:,9]=(tvars[:,:,:,9]!=0)*numpy.log(tvars[:,:,:,9])#"nearTracks_2D_sip" normalized - logscale
    tvars[:,:,:,10]=(tvars[:,:,:,10]!=0)*numpy.log(tvars[:,:,:,10])#"nearTracks_PCAdist" normalized - logscale
    tvars[:,:,:,11]=(tvars[:,:,:,11]!=0)*numpy.log(tvars[:,:,:,11])#"nearTracks_PCAdsig" normalized - logscale
    #"nearTracks_PCAonSeed_x" normalized
    #"nearTracks_PCAonSeed_y" normalized
    #"nearTracks_PCAonSeed_z" normalized
    #"nearTracks_PCAonSeed_xerr" normalized
    #"nearTracks_PCAonSeed_yerr" normalized
    #"nearTracks_PCAonSeed_zerr" normalized
    #"nearTracks_PCAonTrack_x" normalized
    #"nearTracks_PCAonTrack_y" normalized
    #"nearTracks_PCAonTrack_z" normalized
    #"nearTracks_PCAonTrack_xerr" normalized
    #"nearTracks_PCAonTrack_yerr" normalized
    #"nearTracks_PCAonTrack_zerr" normalized
    #"nearTracks_dotprodTrack"
    #"nearTracks_dotprodSeed"
    #"nearTracks_dotprodTrackSeed2D"
    #"nearTracks_dotprodTrackSeed3D"
    #"nearTracks_dotprodTrackSeedVectors2D"
    #"nearTracks_dotprodTrackSeedVectors3D"
    tvars[:,:,:,30]=(tvars[:,:,:,30]!=0)*numpy.log(tvars[:,:,:,30])#"nearTracks_PCAonSeed_pvd" normalized - logscale
    tvars[:,:,:,31]=(tvars[:,:,:,31]!=0)*numpy.log(tvars[:,:,:,31])#"nearTracks_PCAonTrack_pvd" normalized - logscale
    tvars[:,:,:,32]=(tvars[:,:,:,32]!=0)*numpy.log(tvars[:,:,:,32])#"nearTracks_PCAjetAxis_dist" normalized - logscale
    #"nearTracks_PCAjetMomenta_dotprod" 
    tvars[:,:,:,34]=(tvars[:,:,:,34]!=0)*numpy.log(tvars[:,:,:,34])#"nearTracks_PCAjetDirs_DEta" normalized - logscale
    #"nearTracks_PCAjetDirs_DPhi" normalized
    
    #some nan appear due to inversion and log
    tvars=numpy.nan_to_num(tvars)

    if debug: print tvars.shape, svars.shape, jvars.shape, s_rawpt.shape, t_rawpt.shape
        
    return tvars, svars, jvars, s_rawpt, t_rawpt


#basic dataset load and vars producer
sample_file_log="/gpfs/ddn/users/lgiannini/TTbar_samples/test_file.npz"

sample_tvars_log, sample_svars_log, sample_jvars_log , s_rawpt_log, t_rawpt_log= make_vars(sample_file_log)

tcol_mean_log, tcol_std_log = mean_and_std(sample_tvars_log)
scol_mean_log, scol_std_log = mean_and_std(sample_svars_log)

tcol_mean_log2=tcol_mean_log[0,0,:]
tcol_std_log2=tcol_std_log[0,0,:]

scol_mean_log2=scol_mean_log[0,:]
scol_std_log2=scol_std_log[0,:]

scol_mean_log2[17]=0
scol_mean_log2[18]=0
scol_std_log2[17]=1
scol_std_log2[18]=1

def make_varsTransformedNormed(f):
    
    zipfile=numpy.load(f)
    
    jvars=zipfile['arr_0']
    tvars=zipfile['arr_2']
    
    if debug:
        print jvars.shape, "shape of javrs array"
        print tvars.shape, "shape of tavrs array"
        
    tinds = numpy.where(numpy.isnan(tvars))
    tevents_to_remove=numpy.unique(tinds[0])
    
    jinds = numpy.where(numpy.isnan(jvars))
    jevents_to_remove=numpy.unique(jinds[0])
    
    events_to_remove=numpy.unique(numpy.concatenate((tevents_to_remove,jevents_to_remove)))
    events_to_remove=numpy.sort(events_to_remove)
    events_to_remove=events_to_remove[::-1]
    
    if remove:
        for ev in events_to_remove:            
            tvars=numpy.delete(tvars, ev,0)
            jvars=numpy.delete(jvars, ev,0)
            if debug: print "removed event:  ", ev
    else:    
        tvars[tinds]=-10

    tvars=numpy.swapaxes(tvars, 1,2)

    svars=tvars[:,:,0:21] #solo seed_vars
    tvars=tvars[:,:,21:741]#solo quelle utili ovvero nearTracks_
    
    tvars=tvars.reshape((len(tvars),10,20,36))
    
    s_rawpt=svars[:,:,0]
    t_rawpt=tvars[:,:,:,0]
    
    #after raw_pt is assigned--> do this

    if debug: print svars[:,:,0]
    svars[:,:,0]=(svars[:,:,0]!=0)/svars[:,:,0]#"seed_pt"  normalized - logscale    
    if debug: print svars[:,:,0]
    #"seed_eta" normalized
    #"seed_phi" normalized
    #"seed_mass" normalized
    svars[:,:,4]=(numpy.log(abs(svars[:,:,4]))*(svars[:,:,4]!=0)+numpy.min(abs(svars[:,:,4])))*numpy.sign(svars[:,:,4])#"seed_dz" normalized - symmetric log
    svars[:,:,5]=(numpy.log(abs(svars[:,:,5]))*(svars[:,:,5]!=0)+numpy.min(abs(svars[:,:,5])))*numpy.sign(svars[:,:,5])#"seed_dxy" normalized - symmetric log
    svars[:,:,6]=(svars[:,:,6]!=0)*numpy.log(svars[:,:,6])#"seed_3Dip" normalized - log
    svars[:,:,7]=(svars[:,:,7]!=0)*numpy.log(svars[:,:,7])#"seed_3Dsip" normalized - log
    svars[:,:,8]=(svars[:,:,8]!=0)*numpy.log(svars[:,:,8])#"seed_2Dip" normalized - log
    svars[:,:,9]=(svars[:,:,9]!=0)*numpy.log(svars[:,:,9])#"seed_2Dsip" normalized - log
    svars[:,:,10]=(numpy.log(abs(svars[:,:,10]))*(svars[:,:,10]!=0)+numpy.min(abs(svars[:,:,10])))*numpy.sign(svars[:,:,10])#"seed_3DsignedIp" normalized - symmetric log
    svars[:,:,11]=(numpy.log(abs(svars[:,:,11]))*(svars[:,:,11]!=0)+numpy.min(abs(svars[:,:,11])))*numpy.sign(svars[:,:,11])#"seed_3DsignedSip" normalized - symmetric log
    svars[:,:,12]=(numpy.log(abs(svars[:,:,12]))*(svars[:,:,12]!=0)+numpy.min(abs(svars[:,:,12])))*numpy.sign(svars[:,:,12])#"seed_2DsignedIp" normalized - symmetric log
    svars[:,:,13]=(numpy.log(abs(svars[:,:,13]))*(svars[:,:,13]!=0)+numpy.min(abs(svars[:,:,13])))*numpy.sign(svars[:,:,13])#"seed_2DsignedSip" normalized - symmetric log
    #"seed_3D_TrackProbability" -  normalized
    #"seed_2D_TrackProbability" -  normalized
    svars[:,:,16]=(svars[:,:,16]!=0)*numpy.log(svars[:,:,16])#"seed_chi2reduced"  normalized - log
    #"seed_nPixelHits" 
    #"seed_nHits"
    svars[:,:,19]=(svars[:,:,19]!=0)*numpy.log(svars[:,:,19])#"seed_jetAxisDistance" normalized - log
    svars[:,:,20]=(svars[:,:,20]!=0)*numpy.log(svars[:,:,20])#"seed_jetAxisDlength" normalized - log
    
    #some nan appear due to inversion and log
    svars=numpy.nan_to_num(svars)  
    
    tvars[:,:,:,0]=(tvars[:,:,:,0]!=0)/tvars[:,:,:,0]#"nearTracks_pt" normalized - logscale
    #"nearTracks_eta" normalized
    #"nearTracks_phi" normalized
    tvars[:,:,:,3]=(numpy.log(abs(tvars[:,:,:,3]))*(tvars[:,:,:,3]!=0)+numpy.min(abs(tvars[:,:,:,3])))*numpy.sign(tvars[:,:,:,3])#"nearTracks_dz" normalized - symmetric log
    tvars[:,:,:,4]=(numpy.log(abs(tvars[:,:,:,4]))*(tvars[:,:,:,4]!=0)+numpy.min(abs(tvars[:,:,:,4])))*numpy.sign(tvars[:,:,:,4])#"nearTracks_dxy" normalized - symmetric log
    #"nearTracks_mass" normalized
    tvars[:,:,:,6]=(tvars[:,:,:,6]!=0)*numpy.log(tvars[:,:,:,6])#"nearTracks_3D_ip" normalized - logscale
    tvars[:,:,:,7]=(tvars[:,:,:,7]!=0)*numpy.log(tvars[:,:,:,7])#"nearTracks_3D_sip", normalized - logscale
    tvars[:,:,:,8]=(tvars[:,:,:,8]!=0)*numpy.log(tvars[:,:,:,8])#"nearTracks_2D_ip" normalized - logscale
    tvars[:,:,:,9]=(tvars[:,:,:,9]!=0)*numpy.log(tvars[:,:,:,9])#"nearTracks_2D_sip" normalized - logscale
    tvars[:,:,:,10]=(tvars[:,:,:,10]!=0)*numpy.log(tvars[:,:,:,10])#"nearTracks_PCAdist" normalized - logscale
    tvars[:,:,:,11]=(tvars[:,:,:,11]!=0)*numpy.log(tvars[:,:,:,11])#"nearTracks_PCAdsig" normalized - logscale
    #"nearTracks_PCAonSeed_x" normalized
    #"nearTracks_PCAonSeed_y" normalized
    #"nearTracks_PCAonSeed_z" normalized
    #"nearTracks_PCAonSeed_xerr" normalized
    #"nearTracks_PCAonSeed_yerr" normalized
    #"nearTracks_PCAonSeed_zerr" normalized
    #"nearTracks_PCAonTrack_x" normalized
    #"nearTracks_PCAonTrack_y" normalized
    #"nearTracks_PCAonTrack_z" normalized
    #"nearTracks_PCAonTrack_xerr" normalized
    #"nearTracks_PCAonTrack_yerr" normalized
    #"nearTracks_PCAonTrack_zerr" normalized
    #"nearTracks_dotprodTrack"
    #"nearTracks_dotprodSeed"
    #"nearTracks_dotprodTrackSeed2D"
    #"nearTracks_dotprodTrackSeed3D"
    #"nearTracks_dotprodTrackSeedVectors2D"
    #"nearTracks_dotprodTrackSeedVectors3D"
    tvars[:,:,:,30]=(tvars[:,:,:,30]!=0)*numpy.log(tvars[:,:,:,30])#"nearTracks_PCAonSeed_pvd" normalized - logscale
    tvars[:,:,:,31]=(tvars[:,:,:,31]!=0)*numpy.log(tvars[:,:,:,31])#"nearTracks_PCAonTrack_pvd" normalized - logscale
    tvars[:,:,:,32]=(tvars[:,:,:,32]!=0)*numpy.log(tvars[:,:,:,32])#"nearTracks_PCAjetAxis_dist" normalized - logscale
    #"nearTracks_PCAjetMomenta_dotprod" 
    tvars[:,:,:,34]=(tvars[:,:,:,34]!=0)*numpy.log(tvars[:,:,:,34])#"nearTracks_PCAjetDirs_DEta" normalized - logscale
    #"nearTracks_PCAjetDirs_DPhi" normalized
    
    #some nan appear due to inversion and log
    tvars=numpy.nan_to_num(tvars)
    
    tvars=tvars-(tvars!=0)*tcol_mean_log2
    tvars=tvars*(tvars!=0)/tcol_std_log2
    
    svars=svars-(svars!=0)*scol_mean_log2
    svars=svars*(svars!=0)/scol_std_log2

    if debug: print tvars.shape, svars.shape, jvars.shape, s_rawpt.shape, t_rawpt.shape
        
    return tvars, svars, jvars, s_rawpt, t_rawpt


def Drawer(tvars, svars, jvars, s_rawpt, t_rawpt, name="sssss"):
    
    snames=["seed_pt","seed_eta","seed_phi","seed_mass","seed_dz","seed_dxy",
             "seed_3Dip","seed_3Dsip","seed_2Dip","seed_2Dsip","seed_3DsignedIp","seed_3DsignedSip","seed_2DsignedIp","seed_2DsignedSip",
             "seed_3D_TrackProbability" , "seed_2D_TrackProbability", 
             "seed_chi2reduced","seed_nPixelHits","seed_nHits","seed_jetAxisDistance","seed_jetAxisDlength"]

    colour="plum"
    if name=="normed": colour="indigo"
    if name=="logs": colour="b"
    if name=="lognormed": colour="violet"
    
    for i in range(21):
        
        pt_all=s_rawpt
        #pt_all=tvars[:,0,:]

        pt_all=pt_all.reshape((len(pt_all),10))
        pt_all=pt_all.reshape((len(pt_all)*10))

        
        print i, snames[i], name
        #predictions=tvars[:,i:i+1,:]
        predictions=svars[:,:,i:i+1]

        print pt_all.shape
        print predictions.shape
        print len(pt_all)

        predictions=predictions.reshape((len(predictions),10))
        predictions=predictions.reshape((len(predictions)*10))
        print predictions.shape

        plt.cla()
        plt.hist([predictions[pt_all>0]], histtype='bar', stacked=True, bins=100, log=True, color=colour,  edgecolor=colour)
        plt.title(snames[i])
        plt.grid(True, which='both')
        plt.savefig(snames[i]+"_"+name+".png")



    
    tnames=["nearTracks_pt","nearTracks_eta","nearTracks_phi","nearTracks_dz","nearTracks_dxy","nearTracks_mass","nearTracks_3D_ip","nearTracks_3D_sip",
          "nearTracks_2D_ip","nearTracks_2D_sip","nearTracks_PCAdist","nearTracks_PCAdsig","nearTracks_PCAonSeed_x","nearTracks_PCAonSeed_y","nearTracks_PCAonSeed_z",
          "nearTracks_PCAonSeed_xerr","nearTracks_PCAonSeed_yerr","nearTracks_PCAonSeed_zerr","nearTracks_PCAonTrack_x","nearTracks_PCAonTrack_y","nearTracks_PCAonTrack_z",
          "nearTracks_PCAonTrack_xerr","nearTracks_PCAonTrack_yerr","nearTracks_PCAonTrack_zerr","nearTracks_dotprodTrack","nearTracks_dotprodSeed","nearTracks_dotprodTrackSeed2D",
          "nearTracks_dotprodTrackSeed3D","nearTracks_dotprodTrackSeedVectors2D","nearTracks_dotprodTrackSeedVectors3D","nearTracks_PCAonSeed_pvd","nearTracks_PCAonTrack_pvd",
          "nearTracks_PCAjetAxis_dist","nearTracks_PCAjetMomenta_dotprod","nearTracks_PCAjetDirs_DEta","nearTracks_PCAjetDirs_DPhi"]
    
    pt_all=t_rawpt
    print pt_all.shape

    pt_all=pt_all.reshape((len(pt_all),200))
    pt_all=pt_all.reshape((len(pt_all)*200))
    
    colour="gold"
    if name=="normed": colour="yellow"
    if name=="logs": colour="greenyellow"
    if name=="lognormed": colour="lime"
    
    for i in range(36):
        print i
        #predictions=[:,i:i+1,:]
        predictions=tvars[:,:,:,i:i+1]

        print pt_all.shape
        print predictions.shape
        print len(pt_all)

        predictions=predictions.reshape((len(predictions),200))
        predictions=predictions.reshape((len(predictions)*200))
        print predictions.shape

        plt.cla()
        plt.title(tnames[i])
        plt.grid(True, which='both')
        plt.hist([predictions[pt_all>0]], histtype='bar', stacked=True, bins=100, log=True, color=colour,  edgecolor=colour)
        plt.savefig(tnames[i]+"_"+name+".png")

if (debug):

    tvars, svars, jvars, s_rawpt, t_rawpt=make_vars("/gpfs/ddn/users/lgiannini/TTbar_samples/test_file.npz")
    print "drawing 1"
    Drawer(tvars, svars, jvars, s_rawpt, t_rawpt, "regular")

    tvars, svars, jvars, s_rawpt, t_rawpt=make_varsNormalized("/gpfs/ddn/users/lgiannini/TTbar_samples/test_file.npz")
    print "drawing 2"
    Drawer(tvars, svars, jvars, s_rawpt, t_rawpt, "normed")

    tvars, svars, jvars, s_rawpt, t_rawpt=make_varsTransformed("/gpfs/ddn/users/lgiannini/TTbar_samples/test_file.npz")
    print "drawing 3"
    Drawer(tvars, svars, jvars, s_rawpt, t_rawpt, "logs")
    jvars
    tvars, svars, jvars, s_rawpt, t_rawpt=make_varsTransformedNormed("/gpfs/ddn/users/lgiannini/TTbar_samples/test_file.npz")
    print "drawing 4"
    Drawer(tvars, svars, jvars, s_rawpt, t_rawpt, "lognormed")

    tvars, svars, jvars, s_rawpt, t_rawpt=make_vars("/gpfs/ddn/users/lgiannini/TTbar_samples/test_file.npz")
    print "drawing 1"
    Drawer(tvars, svars, jvars, s_rawpt, t_rawpt, "AAA") 

