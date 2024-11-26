import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import cv2
from sklearn.preprocessing import minmax_scale, scale
from math import sqrt
from sklearn import linear_model
from scipy.signal import savgol_filter, butter, filtfilt

# Plot
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["mathtext.default"] = 'regular'

# SKLEARN
from sklearn.preprocessing import minmax_scale, StandardScaler
from tensorflow import keras
from sklearn import linear_model

# Disable TF warning
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


################################################

# INPUT

# PLEASE COMPLETE ALL THE FIELDS BELOW

################################################

# Classifier path
pclass = '/data/CLASSIFIER/' # make sure contains: "Metrics_scaling", "Data_models_NB_Oct14", "ensemble_v16"

# Power Spectra folder:
dpath = '/data/CLASSIFIER_OCT24_ROMAN/data/RC_Cadence_15/'

# global parameters file full path:
dp = '/data/CLASSIFIER_OCT24_ROMAN/results/list_all_RCSimulations_Cadence_15_OS5.psd.globalpars_151024'

# Where to save results? Use final "/"
saveto = '/data/CLASSIFIER_OCT24_ROMAN/test_script/'

# Can change threshold if needed, but default is 0.5
threshold = 0.5

################################################

# INPUT END

################################################


# NN files:
mypath = pclass + 'Data_models_NB_Oct14/'  # SHOULD CONTAIN: shapes_convolved_len300_mod....pkl ; dpfile.pkl
mpath = pclass + 'ensemble_v16/'

# Get name from power spectra folder for saving results
loc = [ match.start() for match in re.finditer('/', dpath)][-2:-1]
name = dpath[loc[0]+1:-1]


if __name__ == '__main__':

    # Read ALL bg corrected time series data, 
    # Delivers DataFrame "spec" with ID, campaign and a dataframe with a line for each star

    entries = sorted(os.listdir(dpath))
    # stri = '.psd.bgcorr'

    entry_list = [ent for ent in entries if ent[-6:]=='bgcorr']
    id_list = []
    for e in entry_list: 
        id_list.append(e[:-11])
        
    ## load individual files 
    spectra = [pd.read_csv(dpath + e, sep='\s+', header=None) for e in entry_list] ## Spectra

    d = {'ID':id_list, 'spectrum':spectra }
    spec = pd.DataFrame(d)
    spec.sort_values(by='ID', inplace=True)
    spec.reset_index(drop=True, inplace=True)
    
    # ----

    if len(id_list)==len(spec):    
        # Read global parameters 
        gp = pd.read_csv(dp , sep='\s+', usecols=range(5))
        gp.columns = ['file','numax','numax_sig','dnu','dnu_sig']
        gp['ID'] = [a[:-4] for a in gp['file']]
    
    # ----

    # Prepare to calculate metrics

    dp = pd.read_pickle(mypath+'dpfile.pkl')
    newl = [np.log10(a) for a in dp.delta_nu3]
    nurg = [0.5*(newl[i]- newl[i+1]) for i in range(len(newl)-1)]
    nurg.append(0)

    ## create cut offs: lower and upper limit of delta nu 
    lims = []
    for i in range(23):
        if i ==0:
            a = 10**(newl[i] + nurg[i])
            b = 10**(newl[i] - nurg[i]) 
        else:
            a = 10**(newl[i] + nurg[i-1])
            b = 10**(newl[i] - nurg[i] )
        lims.append((a,b))
        
    ztest = [0]*23
    for i in range(23):
        ztest[i] = pd.read_pickle(mypath + 'shapes_convolved_len300_mod%i.pkl'%i)

    ########################

    # FUNCTIONS

    # To get the initial short array for folded spectrum 4 delta nu wide
    # This is the same as getlims 8 basically, if we add option n as number of delta nu requested

    ########################
    def getlims8(where_is_nu_max, where_is_dnu, lenxf):
        c = where_is_nu_max - 4*(where_is_dnu)
        
        if c >0:
            a = where_is_nu_max - 4*(where_is_dnu)  # this means there is no conflict on lower limit
            c=0
        else:
            a=0
            b = where_is_nu_max + 4*(where_is_dnu) - c
            #if lower limit is truncated, extend upper limit. This works well if numax is multiple of dnu, c is also mult. of delta nu.                                                                   
        
        # if upper limit was extended, skip next part and go to "return" because there will be no conflict on upper limit 280 Hz.
        
        if c==0: # if high limit was not extended:
            if where_is_nu_max + 4*(where_is_dnu) <lenxf: ## if numax+ 4 dnus fit in spectrum. 
                b = where_is_nu_max + 4*(where_is_dnu)
                ## a stays as left above
            else:## 
                b=lenxf
                a = int(lenxf - 8*(where_is_dnu))
        
        return int(a),int(b)
    ########################


    ########################
    def getlimsold(index_numax, numaxlocal, lendf):
        if numaxlocal>40 and numaxlocal<150:
            lowix = int(0.5*index_numax)
            hiix = int(1.5*index_numax)
        elif numaxlocal>=150 and numaxlocal:
            lowix = int(0.7*index_numax)
            hiix = int(1.3*index_numax) 
        elif numaxlocal<=40:
            lowix = int(0.4*index_numax)
            hiix = int(1.6*index_numax)
        if hiix >= lendf:
            hiix = lendf
        if lowix<=0:
            lowix = 0
        return lowix, hiix
    ########################



    ########################
    # Shift array by n positions
    def shif(arr, n):
        if n!=0:
            e = np.empty_like(arr)
            e[:n] = arr[-n:]
            e[n:] = arr[:-n]
        else:
            e=arr
        return e
    ########################



    ########################
    def autocorr(x):
        result = np.correlate(x, x, mode='full')
        return result[int(result.size/2):]
    ########################


    def acscore2(Y_ACF, modi):    ##no need to plot this by returning 'masked' array, just return score    
        if not Y_ACF.shape == (300,): print('WARNING Wrong Input Shape')   
        wi = 5  ## half width of unmasked portions around .12, .5, 1.,...

        sacv = [np.mean(Y_ACF[4:8]), np.mean(Y_ACF[16+8:50-(wi+8)])]
        sacp = [np.mean(sorted(Y_ACF[8:16], reverse=True)[:2])]  ## mean of top 2 values

        for b in range(1,6,1):  #1,2,3,4,5
            acp = Y_ACF[(b*50)-wi:(b*50)+wi]  ## Y[ 1*50-1: 1*50+1]                                   ## AC peak
            acv = np.mean(Y_ACF[(b*50)+(wi+8):(b+1)*50-(wi+8)])                        ## AC valley
            sacv.append(acv)                                     # mean of the 'valleys'
            sacp.append(np.mean(sorted(acp, reverse=True)[:2]))  # mean of the top 2 points of the peaks.
        
        sco = []
        for a in range(len(sacp)):
            #tmp = sacp[a]/(sacv[a]+sacv[a+1])  ##peaks / average of 'masked' parts on left and right side
            tmp = (sacp[a]-0.5*(sacv[a]+sacv[a+1]))/(0.5*(sacv[a]+sacv[a+1]))
            sco.append(tmp)
        
        if modi>=11: weights = [0.00, 0.00, 1, 0.0, 0.0, 0.00]
        else:        weights = [0.05, 0.3, 0.5, 0.05, 0.1, 0.00]  
        acfscore = sum([a*b for a,b in zip(sco,weights)])
        
        return acfscore


    ########################
    def manhattan_distance(x,y):
        return sum(abs(a-b) for a,b in zip(x,y))
    ########################
    

    # ----------------------------------------------------------------

    # DONT TOUCH BELOW THIS LINE: ALL DATA PROCESS WORK
    ######################
    ######################

    rej = []
    rejnumax = []
    rejdnu = []

    #####################
    nseed= 100 ## choose a seed for repeatability of the results
    ransac = linear_model.RANSACRegressor(random_state= nseed)
    ndnus = 3 ## number of frequency separations to consider
    g = 300 ## N.N. Input data size

    # Dessign butterworth filter
    N  = 4   # Filter order
    Wn = 0.06 # Cutoff frequency
    B, A = butter(N, Wn, output='ba')


    input_ID= []  # 
    input_numax= []  # 
    input_dnu= []  # 
    input_xcor = []
    input_acscore = []
    input_xcor2 = []
    input_Z = []

    id_rej = 0


    for k in range(len(spec)): 


        starID = spec.ID[k]

        ## Full spectrum 
        xf = spec.spectrum[k][0].copy()
        yf = spec.spectrum[k][1].copy()

        ym = yf.max()
        
        ## remove 'unreasonable' single peaks from yf:
        times_mean = 30  ## if a peak is > 30*yf.mean is brought to the next val.
        large_vals = yf[yf>(times_mean*yf.mean())].index.copy()
        max_allowed = yf[yf<=(times_mean*yf.mean())].max()
        for a in large_vals:
            yf[a]=max_allowed


        # indexgp, indexrel
        if gp[gp.ID==spec.ID[k]].shape[0]==0: continue
        indexgp = gp[gp.ID==spec.ID[k]].index.values[0]


        ## Numax and dnu from gp_inmf
        numax_SYD = gp.numax[indexgp]
        dnu_SYD = gp.dnu[indexgp]

        if numax_SYD<=3 or dnu_SYD<=0.3:
            print('negative or too small SYD vals for %s' %starID)
            rej.append(starID)
            rejnumax.append(numax_SYD)
            rejdnu.append(dnu_SYD)
            id_rej = id_rej+1
            continue
        if numax_SYD>=279.8:
            print('too large SYD nu max vals for %is' %starID)
            rej.append(starID)
            rejnumax.append(numax_SYD)
            rejdnu.append(dnu_SYD)
            id_rej = id_rej+1
            continue

        if dnu_SYD>numax_SYD:
            print('Delta nu from SYD is larger than numax for %s' %starID)
            rej.append(starID)
            rejnumax.append(numax_SYD)
            rejdnu.append(dnu_SYD)
            id_rej = id_rej+1
            continue

            
        ## make the frequencies start from zero
        t = xf.iloc[0]
        step = np.diff(xf)[0]

        # Calculate all backward steps in one go
        n_steps = int(np.ceil(t / step))  # Total steps required
        new_xf = np.arange(t - step, 0, -step)[::-1]  # Generate backward values

        # Create new indices for xf and yf
        new_indices = np.arange(-n_steps, 0)

        # Update xf and yf efficiently
        xf = pd.concat([pd.Series(new_xf, index=new_indices), xf])
        yf = pd.concat([pd.Series(0, index=new_indices), yf])

        xf = xf.sort_index().reset_index(drop=True)
        yf = yf.sort_index().reset_index(drop=True)
        
        xfin=xf.copy()
        yfin=yf.copy()
        
        #####
        
        
        ## Get location of parameters
        where_is_nu_max = (xf-numax_SYD).abs().values.argmin()
        numax_local = xf[where_is_nu_max]
        
        where_is_dnu = (xf-dnu_SYD).abs().values.argmin()
        dnu_local = xf[where_is_dnu]
        
        ## Shift nu max to closest multiple of delta nu
        ## To make the position of the modes meaningful
        numax_sh = int((numax_local/dnu_local) + 0.5)*dnu_local
        
        ## Update location of parameters
        where_is_nu_max = (xf-numax_sh).abs().values.argmin()  ##This was changed, now includes shifted numax instead of numax from SYD
        numax_local = xf[where_is_nu_max]


        ## if xf length is not a multiple of dnu, force it by adding zeros at the end
        mod = (len(xf) % (where_is_dnu+1)) ### reminder
        intprev=len(xf) // (where_is_dnu+1) ## floor division
        addtoxf = where_is_dnu+1-mod
        if mod!=0:
            extendxf = np.linspace(xf[-1:].values[0], (intprev+1)*dnu_local, num=addtoxf+1, endpoint=True)
            extendxf = extendxf[1:]
            addto = pd.Series(extendxf)
            # xf = xf.append(addto, ignore_index=True)  ##add extension
            # yf = yf.append(pd.Series([0]*addtoxf), ignore_index=True)  ##add extension
            
            xf = pd.concat([xf, addto], ignore_index=True)  ##add extension
            yf = pd.concat([yf, pd.Series([0]*addtoxf)], ignore_index=True)  ##add extension

            xf.reset_index(drop=True, inplace=True)
            yf.reset_index(drop=True, inplace=True)

        ## if numax + 1 delta nu is >=279uHZ, shift numax left 1 delta nu so there is at least 1 delta nu with data to the right
        if numax_local + dnu_local >=279:
            where_is_nu_max = (xf-(numax_local + dnu_local)).abs().argsort()[:1].values[0] 
            numax_local = xf[where_is_nu_max]


        ## mark where epsilon should be now that we have the spectrum folded correctly
        logtendnu = np.log10(dnu_SYD)
        epsilon = 0.634 + 0.63*logtendnu
        if epsilon>1: epsilon = epsilon-1


        ##  df 8 dnus wide
        a8,b8 = getlims8(where_is_nu_max, where_is_dnu, len(xf))  ## this is the shifted numax

        xcr8 = xf[a8:b8].copy()
        ycr8 = yf[a8:b8].copy()


        ## Change ###
        ### NEW ECHELLE section: from 16 to 11, and from (80,16) to (100,11)
        finspec = (len(yf) // (where_is_dnu+1))
        Z = np.array(yf).reshape(-1,where_is_dnu+1)

        ## Add ~30% of the echelle to the right:
        dim = int((0.3*Z.shape[1])+1)
        Zp = Z[:,:dim]
        ## roll extension 1 step down and replace top line with zeros:
        Zp = np.roll(Zp, -1, axis=0)
        Zp[-1,:]= np.zeros(Zp.shape[1])
        Zext = np.concatenate((Z,Zp),axis=1)


        xext = (np.linspace(0, dnu_local*1.3, num=Zext.shape[1]+1,endpoint=True))  ## this dnu_SYD could be changed to dnu_local to better match everything else
        ye = (np.linspace(0, dnu_local*(Zext.shape[0]) , num=Zext.shape[0]+1,endpoint=True))

        ### Select visible portion:
        f1 = int(numax_SYD//dnu_SYD)
        hl = 5

        if f1 + hl+1 >= finspec:
            Zvis = Zext[-11:]
            yvis = ye[-12:]

        elif f1 - (hl+1) < 2:
            Zvis = Zext[2:13]
            yvis = ye[2:14]

        else:
            Zvis = Zext[(f1-hl):(f1+hl+1)]
            yvis = ye[(f1-hl):(f1+hl+1)+1]

        ## Convert previous section to Standard size array (100x11)
        resZ = cv2.resize(Zvis, dsize=(150, 11), interpolation=cv2.INTER_NEAREST)

        xres = (np.linspace(0, dnu_local*1.3, 150+1,endpoint=True)) # Standard x axis
        yres = yvis


        ## model ## Crosscorrelation
        ######## select appropriate model according to dnu_SYD
        xmod = np.linspace(0,dnu_local,300, endpoint=False)
        for i in range(23):
            if lims[i][0]>=dnu_SYD and dnu_SYD>lims[i][1]:
                zmod = ztest[i].copy()
                modi = i
        zmoddob = np.concatenate((zmod,zmod),axis=0) ## double for better cross-correlation        


        ### Cross-correlation 6 f.s ## FOLDED
        numaxinfs = int((numax_SYD/dnu_local) + 0.5)
        f1 = numaxinfs-3
        f2 = numaxinfs+3
        if f1<1: Z6 = Z[1:7,:]
        elif f2 > finspec: Z6 = Z[-6:,:]
        else: Z6 = Z[f1:f2,:]


        altfol = Z6.sum(axis=0)
        altfol = minmax_scale(altfol)
        xfol = np.linspace(0,dnu_SYD, num=len(altfol), endpoint=True)

        altfol300 = np.interp(xmod, xfol, altfol)
        altxcor = np.correlate(altfol300, zmoddob, 'full') #######################**************
        sh = np.where(altxcor == altxcor.max())[0][0]
        ## shifted folded spectrum
        if sh>300: sh= sh-300
        shzmod = shif(zmod,sh)

        ## V16
        ## smoothing filters
        # Buterworth filter
        smooth_data = filtfilt(B,A, altfol300)
        smooth_datamm = minmax_scale(smooth_data)

        ##########################
        ## Distances between shifted model and filtered folded spec
        scorexc2 = manhattan_distance(shzmod,smooth_datamm) #butter filter

        #################################
        ### Model Similarity Score
        scorexc = (altxcor.max()-np.percentile(altxcor,52))/altxcor.std()

        ## Autocorrelation ##
        ## Go back to original numax value
        ## Cut spectrum between 0.5 numax and 1.5 numax when possible
        where_is_nu_max_in = (xf-numax_SYD).abs().values.argmin()
        numax_local_in = xf[where_is_nu_max_in]

        ## remove unreasonable single peaks from yfin:
        for a in large_vals:
            yfin[a]=max_allowed

        ## Autocorrelation
        a0,b0= getlimsold(where_is_nu_max, where_is_dnu, len(xf)); 
        ycr0 = yfin[a0:b0+1]
        ## avoid zeros in autocorrelation left
        check4zero1 = next((i for i, x in enumerate(ycr0) if x), None) 
        if check4zero1!=0:
            ycr0 = yfin[ a0+check4zero1 : b0+check4zero1 ]
        ## avoid zeros in autocorrelation right
        check4zero2 = next((i for i, x in enumerate(reversed(ycr0.to_list())) if x), None)
        if check4zero2!=0:
            ycr0 = yfin[ a0-check4zero2 : b0-check4zero2 ]
        tmp = autocorr(ycr0)[:ndnus*(where_is_dnu)+1].copy()
        tmp.resize(ndnus*(where_is_dnu)+1)
        if numax_SYD <7: val3 = tmp[3]; tmp[:3] = [val3]*3
        else: 
            try:
                val6 = tmp[6]; tmp[:6] = [val6]*6
            except IndexError:
                print('Wrong autocorrelation length for %is' %starID)
                rej.append(starID)
                rejnumax.append(numax_SYD)
                rejdnu.append(dnu_SYD)
                id_rej = id_rej+1
                continue

        Y_ACF_0 = minmax_scale(tmp)
        X_ACF_0 = (np.arange(3*where_is_dnu+1)/(where_is_dnu))


        # ### RASNAC Regression Fit
        # ## Reshaping to use with rasnac fit
        Y_ransac = np.array(Y_ACF_0).reshape(-1,1)
        X_ransac= X_ACF_0.reshape(-1,1)

        try: ransac.fit(X_ransac, Y_ransac); line_y_ransac = ransac.predict(X_ransac)
            # Predict data of estimated models and substract it from ACF
        except ValueError: print('could not fit ransac, ID %s' %starID);line_y_ransac = 0

        # Predict data of estimated models and substract it from ACF
        Y_ACF_1 = Y_ransac - line_y_ransac

        # Reduce number of data points,  and interpolate
        X_ACF = np.linspace(X_ACF_0.min(), X_ACF_0.max(), g) 
        Y_ACF= np.interp(X_ACF , X_ACF_0.reshape(-1), Y_ACF_1.reshape(-1)) ## result has length=g

        # Bring all values between 0,1
        Y_ACF = minmax_scale(Y_ACF)
        ### Make sure no nans
        Y_ACF =np.nan_to_num(Y_ACF)
        # Auto-Correlation score
        scoreac = acscore2(Y_ACF, modi) 

    # save inputs for NN

        input_Z.append(resZ)
        input_ID.append(starID)
        input_numax.append(numax_SYD)
        input_dnu.append(dnu_SYD)
        input_acscore.append(scoreac)
        input_xcor.append(scorexc)
        input_xcor2.append(scorexc2)


    tmp1 = pd.DataFrame()
    tmp1.insert(0,'ID', input_ID)
    tmp1.insert(1,'numax', input_numax)
    tmp1.insert(2,'dnu', input_dnu )
    tmp1.insert(3,'scoreAC', input_acscore )
    tmp1.insert(4,'scoreXC', input_xcor )
    tmp1.insert(5,'scoreXC2', input_xcor2)
    tmp1.insert(6,'echelle', input_Z )
    tmp1.insert(7,'det', [-2]*len(input_ID) )

    for a,b,c in zip(rej, rejnumax, rejdnu):
        tmp1.loc[tmp1.shape[0]] = [a, b,c,0,0,0,0,0]  ## adding rejected IDs to last rows in dataframe

    # tmp1.to_pickle(pikledname)
    # ---------------------------------------------------------------- NOT NECESSARY TO SAVE THIS FILE, WILL USE IT NEXT AND NOT NEEDED AFTER THAT


    #################
    # BEGINS NN
    #################


    # standardize the dataset
    metrics_scaler = pd.read_pickle(pclass + 'Metrics_scaling')

    tsamp = tmp1.copy()

    ## Models
    mpath = pclass + 'ensemble_v16/'
    models = sorted(os.listdir(mpath))
    mlist = [ent for ent in models if ent.startswith('MAR')]

    tsamp2 = tsamp
    tsamp = tsamp[tsamp.det!=0] 

    AC = (tsamp.scoreAC-metrics_scaler['scoreAC']['mean'])/metrics_scaler['scoreAC']['std']
    XC1 = (tsamp.scoreXC-metrics_scaler['scoreXC']['mean'])/metrics_scaler['scoreXC']['std']
    XC2 = (tsamp.scoreXC2-metrics_scaler['scoreXC2']['mean'])/metrics_scaler['scoreXC2']['std']

    X_scaled = pd.DataFrame({'scoreAC': AC, 'scoreXC': XC1, 'scoreXC2': XC2})

    ## Categorical metric:

    bins = np.logspace(np.log10(14),np.log10(280), num=6)
    numax = tsamp['numax'].copy()
    NM1 = []
    for a in numax:
        if a<=bins[0]:
            tmp = [1,0,0,0,0,0]
        elif (a>bins[0]) and (a<=bins[1]):
            tmp = [0,1,0,0,0,0]
        elif (a>bins[1]) and (a<=bins[2]):
            tmp = [0,0,1,0,0,0]
        elif (a>bins[2]) and (a<=bins[3]):
            tmp = [0,0,0,1,0,0]
        elif (a>bins[3])and (a<=bins[4]):
            tmp = [0,0,0,0,1,0]
        elif (a>bins[4]):
            tmp = [0,0,0,0,0,1]
        NM1.append(tmp)
    NM = pd.DataFrame(np.row_stack(NM1))
        

    metrics = X_scaled.join(NM)
        
    echar = tsamp.echelle.copy()
    ech = np.stack([a for a in echar])

    # ---------------------------
    # Load 40 models
    # ---------------------------

    predbym = pd.DataFrame({'ID':tsamp.ID})

    k=1
    for a in mlist:
        mm = keras.models.load_model(mpath+a)

        pvals = mm.predict([ech, metrics]) 
        predval = [b[0] for b in pvals]

        predbym.insert(k, a, predval)
        k+=1   

    consensus =  (predbym.iloc[:,1:].sum(axis=1))/(predbym.shape[1]-1)
    predbym.insert(1, 'consensus', consensus)


    # HISTOGRAM
    a = predbym.consensus.hist(bins=20, color='k', edgecolor='silver'); 
    a.set_xlabel('Probability')
    a.set_ylabel('Number')

    fig = a.get_figure()
    fig.savefig(saveto + 'histogram_'+ name + '.png')

    # PREPARE RESULTS FILE

    df = pd.DataFrame({'ID':tsamp.ID, 'pred': predbym.consensus, 'numax': tsamp.numax, 'dnu': tsamp.dnu})
    tmp = [0 if a<=0.5 else 1 for a in df.pred ]
    df.insert(1,'good_dnu', tmp)
    df['ID'] = [a for a in df.ID]

    # adding at the end of the table those stars that we couldnt test for being out of any of the constraints
    for a in tsamp2[tsamp2.det==0].index:
        df.loc[df.shape[0]] = [tsamp2.ID[a], 0,0, tsamp2.numax[a], tsamp2.dnu[a]]

    ALL = df.shape[0]
    GOOD = df[df.good_dnu==1.0].shape[0]

    # Results plot

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,5), dpi=100)

    dd = df[df.pred>=threshold].reset_index()
    dr = df[df.pred<threshold].reset_index()

    ## scatter rejected
    ax.scatter(dr.numax, dr.dnu, c='grey', s=3, alpha=0.3, label='No %i'%len(dr))
    ## Scatter accepted
    sc = ax.scatter(dd.numax , dd.dnu , c='lime', alpha=0.8, s=3, label='Yes %i'%len(dd))

    ax.set_title(name + ' - Threshold %.2f'%threshold)
    ## Ref lines
    x=np.linspace(3,df.numax.max(), num=100)
    y = 0.27*(x**0.77)
    [plt.plot(x,h, color=c, linestyle=l, alpha=0.7, label=lb) for h,c,l,lb  in zip([y, y*1.35, 0.75*y], ['crimson', 'k', 'k'], ['-', '--', '-.'], ['$y=0.27*(x^{0.77})$', '$1.35*y$', '$0.75*y$']) ]

    ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(which='both', alpha=0.3)
    ax.set_xlabel('$\\nu_{max}$'); ax.set_ylabel('$\\Delta \\nu$'); ax.legend()

    fig = ax.get_figure()
    fig.savefig(saveto + 'results_'+ name + '.png')

    # summary of results
    fp1 = df[(df.dnu<0.75*0.267*df.numax**0.764) & (df.good_dnu==1)].index.tolist()
    fp2 = df[(df.dnu>1.35*0.267*df.numax**0.764) & (df.good_dnu==1)].index.tolist()
    fp = fp1 + fp2

    tmp = [1 if a in fp else 0 for a in df.index]
    df['fp'] = tmp

    df.to_csv(saveto + name + '_ALL_%i_GOODDNU_%i_FP_%i'%(ALL, GOOD, len(fp)), index=False)

