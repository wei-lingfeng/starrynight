import numpy as np
import matplotlib.pyplot as plt
import os, sys
from astropy.io import fits
from astropy.table import Table
from scipy.signal import find_peaks
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit


############################################

# clickpoints = []
# def onclick(event):
#     print(event)
#     global clickpoints
#     clickpoints.append([event.ydata])
#     #print(clickpoints)
#     ax1.axhline(event.ydata, c='r', ls='--')
#     ax2.axhline(event.ydata, c='r', ls='--')
#     plt.draw()
#     if len(clickpoints) == 2:
#         print('Closing Figure')
#         plt.draw()
#         plt.pause(1)
#         plt.close('all')

# def onclickclose(event):
#     plt.close('all')

def NormDist(x, mean, sigma, baseline, amplitude):
    return amplitude * 1. / np.sqrt(2. * np.pi * sigma**2) * np.exp(-1*(x - mean)**2 / (2.*sigma**2) ) + baseline

def twoDist(x, mean1, sigma1, baseline1, amplitude1,
            mean2, sigma2, baseline2, amplitude2):
    Dist = amplitude1 * 1. / np.sqrt(2. * np.pi * sigma1**2) * np.exp(-1*(x - mean1)**2 / (2.*sigma1**2) ) + baseline1 + \
        amplitude2 * 1. / np.sqrt(2. * np.pi * sigma2**2) * np.exp(-1*(x - mean2)**2 / (2.*sigma2**2) ) + baseline2
    return Dist

def twoDist2(x, mean1, sigma, baseline, amplitude1,
                mean2, amplitude2):
    Dist = amplitude1 * 1. / np.sqrt(2. * np.pi * sigma**2) * np.exp(-1*(x - mean1)**2 / (2.*sigma**2) ) + baseline + \
        amplitude2 * 1. / np.sqrt(2. * np.pi * sigma**2) * np.exp(-1*(x - mean2)**2 / (2.*sigma**2) )
    return Dist

def lnlike(theta, x, y):
    mean1, sigma, baseline, amplitude1, mean2, amplitude2 = theta
    model = amplitude1 * 1. / np.sqrt(2. * np.pi * sigma**2) * np.exp(-1*(x - mean1)**2 / (2.*sigma**2) ) + baseline + \
            amplitude2 * 1. / np.sqrt(2. * np.pi * sigma**2) * np.exp(-1*(x - mean2)**2 / (2.*sigma**2) ) + baseline
    return -0.5*(np.sum((y-model)**2))

def lnlike2(theta, x, y):
    mean1, sigma1, baseline1, amplitude1, mean2, sigma2, baseline2, amplitude2 = theta
    model = amplitude1 * 1. / np.sqrt(2. * np.pi * sigma1**2) * np.exp(-1*(x - mean1)**2 / (2.*sigma1**2) ) + baseline1 + \
            amplitude2 * 1. / np.sqrt(2. * np.pi * sigma2**2) * np.exp(-1*(x - mean2)**2 / (2.*sigma2**2) ) + baseline2
    return -0.5*(np.sum((y-model)**2))



#########################################
## Parameters Set Up
#########################################
date = (15, 12, 24) # maximum 2-digit year, month, day
sci_frames = [40, 41, 42, 43]
orders = [32, 33]
diagnosticplot = True
user_confirm = True

year, month, day = date
year    = str(year).zfill(2)
month   = str(month).zfill(2)
day     = str(day).zfill(2)

month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

for sci_frame in sci_frames:
    for order in orders:
        
        if int(year) > 18:
            # For data after 2018, sci_names = [nspec200118_0027, ...]
            sci_name = 'nspec' + year + month + day + '_' + str(sci_frame).zfill(4)

        else:
            # For data prior to 2018 (2018 included)
            sci_name = month_list[int(month) - 1] + day + 's' + str(sci_frame).zfill(4)
        
        print('science name: {}'.format(sci_name))
        print('order: {}'.format(order))
        
        # sci data path
        common_path = '/home/l3wei/ONC/Data/20{}{}{}/reduced'.format(year, month_list[int(month)-1], day)
        data_path  = common_path + '/nsdrp_out/fits/order'
        data_pathN = common_path + '/nsdrp_out/fits/noiseorder'
        data_pathT = common_path + '/nsdrp_out/fits/all'

        # save to path
        save_to_path = common_path + '/extracted_binaries/%s/O%s/'%(sci_name, order)
        print(save_to_path)

        if not os.path.exists(save_to_path):
            os.makedirs(save_to_path)

        ############################################
        

        hdul = fits.open(data_path + '/%s_%s_order.fits'%(sci_name, order))
        #print(hdul.info())
        data = hdul['PRIMARY'].data
        #print(data)
        #print(data.shape)

        # Integration time
        itime = hdul['PRIMARY'].header['ITIME'] # in seconds

        # Noise image
        hdul2 = fits.open(data_pathN + '/%s_%s_noiseorder.fits'%(sci_name, order))
        #print(hdul2.info())
        noise = hdul2['PRIMARY'].data
        halfway = data.shape[0]/2.


        '''
        #print(noise)
        #print(noise.shape)
        #print(data[17:21,:].shape)
        print('object:', np.sum(data[17:21,:], axis=0))
        print('object noise:,', np.sum(noise[17:21,:], axis=0))
        print('sky noise:,', np.median(noise[30:40,:], axis=0))
        print('sky:', np.median(data, axis=0))
        #sys.exit()

        plt.figure(figsize=(10,5))
        plt.imshow(data, aspect='auto', origin='lower')
        plt.axhline(17, color='r', ls='--')
        plt.axhline(20, color='r', ls='--')


        plt.figure(figsize=(10,5))
        plt.plot(np.sum(data[17:21,:], axis=0))
        plt.plot(np.sum(noise[17:21,:], axis=0))
        #plt.axhline(17, color='r', ls='--')
        #plt.axhline(20, color='r', ls='--')

        plt.show()
        sys.exit()
        '''

        fig1 = plt.figure(figsize=(10,5))
        # cid  = fig1.canvas.mpl_connect('button_press_event', onclick)
        ax1  = fig1.add_subplot(121)
        ax2  = fig1.add_subplot(122) 

        ax1.imshow(data, aspect='auto', origin='lower')

        Y = np.sum(data, axis=1)
        X = range(len(Y))

        Y1 = data[:,5]
        X1 = range(len(Y))

        ax2.plot(Y, X, c='C0')
        #plt.plot(Xs, twoDist(Xs, *result['x']), c='r', alpha=0.5)
        ax2.set_ylim(np.min(X1), np.max(X1))
        ax1.set_ylabel('Pixel')
        ax1.set_xlabel('Pixel')
        ax2.set_xlabel('Counts')
        ax1.minorticks_on()
        ax2.minorticks_on()
        plt.suptitle('Click on either image to close to the peak pixels to select both componenets of the binary')
        plt.show()
        plt.close('all')


        # find highest two peaks
        peaks, properties = find_peaks(Y, height=0)
        peaks = peaks[np.argsort(properties['peak_heights'])[[-1, -2]]]
        print('Peak pixels: {}'.format(peaks))

        fig, ax = plt.subplots()
        ax.plot(Y)
        ax.scatter(peaks, Y[peaks])
        ax.vlines(peaks, min(Y), Y[peaks], colors='C3', linestyles='dashed')
        ax.set_xlabel('pixel')
        ax.set_ylabel('count')
        plt.show()
        
        pixel1, pixel2 = peaks
        
        
        if user_confirm:
            
            confirm = input('Confirm? (y/n)\n')
            
            while confirm not in ['y', 'n']:
                print("Input not recognized. Please in put 'y' or 'n'.\n")
                confirm = input('Confirm? (y/n)\n')
            
            # continue to manually input until correct:
            if confirm == 'n':
                while True:
                    pixel1, pixel2 = [int(_) for _ in input('Please input pixel1 and pixel2:\n').strip().split(', ')]
                    fig, ax = plt.subplots()
                    ax.plot(Y)
                    ax.scatter([pixel1, pixel2], Y[[pixel1, pixel2]])
                    ax.vlines([pixel1, pixel2], min(Y), Y[[pixel1, pixel2]], colors='C3', linestyles='dashed')
                    ax.set_xlabel('pixel')
                    ax.set_ylabel('count')
                    plt.show()
                    
                    confirm = input('Confirm? (y/n)\n')
                    if confirm == 'y':
                        break
            
            elif confirm == 'y':
                pass
                
         

        # # Which one is the primary/secondary?
        
        counts1 = Y[pixel1]
        counts2 = Y[pixel2]
        
        if counts1 > counts2:
            primaryPix, primaryCounts     = pixel1, counts1
            secondaryPix, secondaryCounts = pixel2, counts2

        else:
            primaryPix, primaryCounts     = pixel2, counts2
            secondaryPix, secondaryCounts = pixel1, counts1

        # mean1, sigma, baseline, amplitude1, mean2, amplitude2
        p0 = [primaryPix, 2, np.median(Y), primaryCounts, secondaryPix, secondaryCounts]
        bounds = [[p0[0]-2, p0[1]-0.5, p0[2]*0.9, 0,     p0[4]-2, 0     ],
                [p0[0]+2, p0[1]+0.5, p0[2]*1.1, p0[3], p0[4]+2, p0[5]]]
        #p0 = [21, 2, 50000, 800000, 15, 2, 300000]
        #p0 = [21, 2, 50000, 3e6, 15, 2, 50000, 9e5]

        # popt: mean1, sigma, baseline, amplitude1, mean2, amplitude2
        popt, pcov = curve_fit(twoDist2, X, Y, p0=p0)
        # bounds = [[popt[0]-2, popt[1]-0.5, popt[2]*0.9, 0,       popt[4]-2, 0      ],
        #           [popt[0]+2, popt[1]+0.5, popt[2]*1.1, popt[3], popt[4]+2, popt[5]]]
        print('Optimal parameters: {}',format(popt))
        poptorig = popt

        fig, ax1 = plt.subplots(figsize=(7,5))
        # cid = fig.canvas.mpl_connect('button_press_event', onclickclose)
        ax1.plot(X, Y, c='b', label='data')
        Xs = np.linspace(np.min(X), np.max(X), 10000)
        ax1.plot(Xs , twoDist2(Xs, *popt), c='r', alpha=0.5, label='fit')
        #plt.plot(Xs, twoDist(Xs, *result['x']), c='r', alpha=0.5)
        ax1.set_xlim(np.min(X), np.max(X))
        ax1.legend()
        ax1.set_ylabel('Counts')
        ax1.set_xlabel('Pixel')
        ax1.set_title('Click on image to close')
        ax1.minorticks_on()
        plt.tight_layout()
        plt.show()
        plt.close('all')

        print('Running...')
        sys.stdout.flush()
        
        if diagnosticplot: 
            pdf = PdfPages(save_to_path+'%s_BinaryFits.pdf'%sci_name)

        FluxesA = []
        FluxesB = []
        NoisesA = []
        NoisesB = []
        Pixels  = []
        Sigmas  = []
        testcount = 0
        for i in range(data.shape[1]):
            # sys.stdout.write("Pixel: %s/%s"%(i, data.shape[1]) + '\r')
            # sys.stdout.flush()
            #if testcount==4: sys.exit()
            try:
                Y = data[:,i]
                X = range(len(Y))
                
                # popt, pcov = curve_fit(twoDist2, X, Y, p0=popt, bounds=bounds, maxfev=100000, method='dogbox')
                peaks, properties = find_peaks(Y, height=0)
                peaks = peaks[np.argsort(properties['peak_heights'])[[-1, -2]]]
                
                pixel1, pixel2 = peaks
                
                counts1 = Y[pixel1]
                counts2 = Y[pixel2]
                
                # mean1, sigma, baseline, amplitude1, mean2, amplitude2
                p0 = [pixel1, 2, np.median(Y), counts1, pixel2, counts2]
                
                popt, pcov = curve_fit(twoDist2, X, Y, p0=p0)
                
                #if i < 5 and data.shape[1] == 2048: popt = poptorig # dirty fix for the upgraded nirspec
                
                
                # popt: mean1, sigma, baseline, amplitude1, mean2, amplitude2
                # NormDist(x, mean, sigma, baseline, amplitude)
                fluxA     = np.trapz(NormDist(Xs, popt[0], popt[1], 0, popt[3]), x=Xs) / itime
                fluxB     = np.trapz(NormDist(Xs, popt[4], popt[1], 0, popt[5]), x=Xs) / itime

                # DOES THIS NEED TO BE WEIGHTED BY THE ITIME?
                #print(np.floor(popt[0]-2*popt[1]), np.ceil(popt[0]+2*popt[1])+1)
                NoiseSqrA = np.sum(noise[int(np.floor(popt[0]-2*popt[1])):int(np.ceil(popt[0]+2*popt[1]))+1,i]) 
                NoiseSqrB = np.sum(noise[int(np.floor(popt[4]-2*popt[1])):int(np.ceil(popt[4]+2*popt[1]))+1,i]) 

                TotPix    = popt[1]*4

                #Sky       = (popt[2]*TotPix - np.median(popt[2]*TotPix)) / itime
                if popt[2] < halfway:
                    SkyNoise = np.sum(noise[int(np.floor(popt[0]+int(halfway/2)-2*popt[1])):int(np.ceil(popt[0]+int(halfway/2)+2*popt[1]))+1,i]) 
                elif popt[2] >= halfway:
                    SkyNoise = np.sum(noise[int(np.floor(popt[0]-int(halfway/2)-2*popt[1])):int(np.ceil(popt[0]-int(halfway/2)+2*popt[1]))+1,i]) 
                    #SkyNoise  = ((popt[2]-int(halfway/2))*TotPix) / itime

                #NoiseA    = np.sqrt(NoiseSqrA + Sky)
                #NoiseB    = np.sqrt(NoiseSqrB + Sky)
                NoiseA    = np.sqrt(NoiseSqrA + SkyNoise) / itime
                NoiseB    = np.sqrt(NoiseSqrB + SkyNoise) / itime

                #print(itime, TotPix, fluxA * itime, NoiseSqrA * itime, SkyNoise * itime, NoiseA)
                testcount+=1

                FluxesA.append(fluxA)
                NoisesA.append(NoiseA)
                FluxesB.append(fluxB)
                NoisesB.append(NoiseB)
                Pixels.append(i)
                Sigmas.append(popt[1])

                if diagnosticplot:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    ax.plot(X, Y)
                    #print(popt)
                    #print('FWHM (pixels): %s'%(popt[1]*2.355))
                    #result = op.minimize(nll2, p0, args=(X, Y), method='Nelder-Mead')
                    #print(result['x'])
                    Xs = np.linspace(np.min(X), np.max(X), 10000)
                    ax.plot(Xs, twoDist2(Xs, *popt), c='r', alpha=0.5)
                    #plt.plot(Xs, twoDist(Xs, *result['x']), c='r', alpha=0.5)
                
                    at = AnchoredText('mu1: %0.2f \n sigma: %0.2f \n baseline: %0.2f \n amplitude1: %0.2f \n mu2: %0.2f \n amplitude2: %0.2f'%(popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]),
                                    prop=dict(size=10), frameon=False,
                                    loc=1,
                                    )

                    ax.add_artist(at)
                    plt.title('Page: %s'%(i+1))
                    pdf.savefig()
                    plt.close()
                    #plt.show()
                    #sys.exit()
            except:
                FluxesA.append(np.nan)
                NoisesA.append(np.nan)
                FluxesB.append(np.nan)
                NoisesB.append(np.nan)
                Pixels.append(i)
                Sigmas.append(i)
                continue

        if diagnosticplot:

            fig        = plt.figure()
            ax         = fig.add_subplot(111)
            Sigmas     = np.array(Sigmas)
            plotSigmas = Sigmas[0:1010][np.where(Sigmas[0:1010] > 0)] # clip bad points and end points
            ax.hist(plotSigmas, bins=int(len(plotSigmas)))
            ax.axvline(np.mean(plotSigmas), c='r', ls='--')
            ax.axvline(np.mean(plotSigmas)+np.std(plotSigmas), c='r', ls=':')
            ax.axvline(np.mean(plotSigmas)-np.std(plotSigmas), c='r', ls=':')
            plt.title('Sigmas: mu=%0.3f, sigma=%0.3f'%(np.mean(plotSigmas), np.std(plotSigmas)))
            pdf.savefig()
            plt.close()
            pdf.close()

        # Which one is the primary/secondary
        if np.nansum(FluxesA) >= np.nansum(FluxesB):
            PrimaryFlux    = FluxesA
            PrimaryNoise   = NoisesA
            SecondaryFlux  = FluxesB
            SecondaryNoise = NoisesB
        else:
            PrimaryFlux    = FluxesB
            PrimaryNoise   = NoisesB
            SecondaryFlux  = FluxesA
            SecondaryNoise = NoisesA


        t1 = Table([Pixels, PrimaryFlux, PrimaryNoise, SecondaryFlux, SecondaryNoise, Sigmas], names=['pixel', 'primary_flux', 'primary_noise', 'secondary_flux', 'secondary_noise', 'sigmas'])
        t1.write(save_to_path + 'extracted_spectra.csv', overwrite=True)

        # Save the spectra
        plt.figure(figsize=(10,6))
        plt.plot(Pixels, PrimaryFlux, label='Flux')
        plt.plot(Pixels, PrimaryNoise, label='Noise')
        plt.legend(frameon=False)
        plt.minorticks_on()
        plt.xlabel('Pixel')
        plt.ylabel('Counts/s')
        plt.savefig(save_to_path + 'Primary_Spectrum.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(Pixels, SecondaryFlux, label='Flux')
        plt.plot(Pixels, SecondaryNoise, label='Noise')
        plt.minorticks_on()
        plt.legend(frameon=False)
        plt.xlabel('Pixel')
        plt.ylabel('Counts/s')
        plt.savefig(save_to_path + 'Secondary_Spectrum.png', dpi=300, bbox_inches='tight')
        plt.close()

        fullpath   = data_pathT + '/' + sci_name + '_' + str(order) + '_all.fits'
        save_nameA = save_to_path + '%s_A_%s_all.fits'%(sci_name, order)
        save_nameB = save_to_path + '%s_B_%s_all.fits'%(sci_name, order)

        hdulistA         = fits.open(fullpath)
        hdulistA[1].data = np.array(PrimaryFlux)
        hdulistA[2].data = np.array(PrimaryNoise)
        hdulistA[1].header['COMMENT']   = 'Primary Extracted Spectrum'

        hdulistB         = fits.open(fullpath)
        hdulistB[1].data = np.array(SecondaryFlux)
        hdulistB[2].data = np.array(SecondaryNoise)
        hdulistB[1].header['COMMENT']   = 'Secondary Extracted Spectrum'

        try:
            hdulistA.writeto(save_nameA, overwrite=True)
            hdulistB.writeto(save_nameB, overwrite=True)
        except FileNotFoundError:
            hdulistA.writeto(save_nameA)
            hdulistB.writeto(save_nameB)

        print('Finished!\n\n')