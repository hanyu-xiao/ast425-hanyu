"""
polarization.py combines the Q and U images produced by wsclean into a MFS P image and writes it to a FITS file, as well as a polarization fraction image.

example use case:
python polarization.py --pathQU 'sb44780/imageQU/SB44780-' --pathI 'sb44780/image/SB44780-' --channels 72 --saveto 'sb44780/results/SB44780-' --doDiagnostics --doWrite --correctForPBR

We assume polarized intensity to noise ratio < 4. (George et al. 2012)
We find polarization fraction at the location of highest polarized intensity. This is not necessarily the location of highest polarization fraction.
Script took 3 minutes to run with 72 channels and 3240x3240 images on an average laptop.
"""

# constants and import
import argparse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
from scipy.ndimage import label
from uncertainties import unumpy

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Combines the Q and U images produced by wsclean into P images.")
    parser.add_argument("--pathQU", required=True, help="If a file path to an image is 'sb44780/SB44780QU-0000-Q-image.fits' then set pathQU='sb44780/SB44780QU'")
    parser.add_argument("--pathI", required=True, help="Provide the full file path to the MFS I image FITS file'")
    parser.add_argument("--channels", type=int, required=True, help="Specify number of channels used in wsclean")
    parser.add_argument("--saveto", required=True, help="If a desired file path is 'sb44780/SB44780-MFS-P-image.fits' then set saveto='sb44780/SB44780'")
    parser.add_argument("--pathCat", default="consolidated_catalog_ver1.2.0.fits", help="Path to consolidated catalog")
    parser.add_argument("--emptyRegionRange", default="slice(640, 860), slice(980, 1390)", help="Region to calculate noise from")
    parser.add_argument("--SBID", default="44780", help="Source ID for diagnostics")
    parser.add_argument("--saveChannelP", action="store_true", help="Save the polarized intensity image for each channel")
    parser.add_argument("--doDiagnostics", action="store_true", help="Calculate and produce diagnostic plots")
    parser.add_argument("--doWrite", action="store_true", help="Write output to FITS file")
    parser.add_argument("--correctForPBR", action="store_true", help="Correct for primary beam response")
    parser.add_argument("--PSNR", type=float, default=5, help="Minimum SNR for polarized intensity masking")
    parser.add_argument("--ISNR", type=float, default=100, help="Minimum SNR for total intensity masking")
    parser.add_argument("--antennaDiameter", type=float, default=12, help="Antenna diameter in meters for FWHM calculation")
    return parser.parse_args()

def calculate_polarization(args):
    """Main function to calculate polarization images."""
    # Parse additional parameters
    emptyRegionRange = eval(args.emptyRegionRange)
    
    # read in Stokes I MFS image and Header
    with fits.open(f'{args.pathI}MFS-image.fits') as hdulI:
        dataI = hdulI[0].data[0,0] # Change the shape from (1,1,n,n) to (n,n)
        headerI = hdulI[0].header
    degPerPix = abs(headerI['CDELT1']) # degrees per pixel of image, assuming it is same for both axes
    imageLenX = headerI['NAXIS1'] # number of pixels along this axis
    imageLenY = headerI['NAXIS2'] # these might be reversed, but the image is square so it's fine for this case
    x = np.arange(-imageLenX // 2, imageLenX // 2)
    y = np.arange(-imageLenY // 2, imageLenY // 2)
    x, y = np.meshgrid(x, y)
    obsFreq = headerI['CRVAL3']

    if args.correctForPBR:
        obsFWHM = np.degrees(1.09 * c / obsFreq / args.antennaDiameter) # coefficient from https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S1323358020000417
        obsPBRsigma = obsFWHM / 2.35482004503 # coefficient from https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/
        obsPBR =  np.exp(-(x**2 + y**2) / (2 * (obsPBRsigma/degPerPix)**2))
        obsPBR /= obsPBR.max()

        dataI /= obsPBR # correct the Stokes I MFS image for primary beam response

    # mask dataI values below SNR of ISNR
    Imask = np.where(dataI < args.ISNR * dataI[emptyRegionRange].std(), np.nan, 1)
    dataImasked = dataI * Imask
    # identify clusters (bright sources) in the Stokes I image
    labeled_array, num_clusters = label(~np.isnan(Imask))

    if args.doDiagnostics:
        # read in pol catalogue
        polCat = Table.read(args.pathCat)

        # Define the WCS of the image
        wcs = WCS(headerI).celestial

        # Define the pixel coordinates of the image edges
        imageRangeX = np.array([0, imageLenX])
        imageRangeY = np.array([0, imageLenY]) 

        # Convert pixel coordinates to world coordinates (RA, Dec) in degrees
        [raMax, raMin], [decMin, decMax] = wcs.all_pix2world(imageRangeX, imageRangeY, 0)

        # Filter the table
        # ra was negative, we add 360 to get it in the 0-360 range
        mask = ((polCat['ra'] >= raMin + 360) & (polCat['ra'] <= raMax + 360) & (polCat['dec'] >= decMin) & (polCat['dec'] <= decMax))
        polCatMasked = polCat[mask]

        # Convert RA and Dec of polarized sources to pixel coordinates
        catSourceXpixs, catSourceYpixs = wcs.all_world2pix(polCatMasked['ra'], polCatMasked['dec'], 0)

        # Read Stokes I, polarization intensity, polarization fraction, frequency, and rotation measure from catalog
        catSourceIs = polCatMasked['stokesI']
        catSourcePs = polCatMasked['polint']
        catSourcePolFracs = polCatMasked['fracpol']
        catSourceFreqs = np.array([(1435100000.0+1364900000.0)/2, (1435100000.0+1364900000.0)/2, 1420780900.0])/1e0 #FIXME applicable only to this specific case
        catSourceRMs = polCatMasked['rm'] # units of radians per metre squared
        catSourceNames = ['J1911+1923', 'J1913+2019'] 
        # Catalog sources are matched to clusters by eye. This can be automated but would scale O(n^2) with catalog length.
        beamCatalogPairIDs = [16,55] if not args.correctForPBR else [23,66] # some sources now pass the threshold

    # Initialize per channel arrays
    channelPs = np.empty((args.channels, imageLenX, imageLenY))
    channelFreqs = np.empty(args.channels)
    channelWavelengths = np.empty(args.channels)
    if args.doDiagnostics:
        channelPolAngles = np.empty((len(beamCatalogPairIDs), args.channels))
        channelSourceIs = np.empty((len(beamCatalogPairIDs), args.channels))
        channelSourcePs = np.empty((len(beamCatalogPairIDs), args.channels))
        channelSourceQs = np.empty((len(beamCatalogPairIDs), args.channels))
        channelSourceUs = np.empty((len(beamCatalogPairIDs), args.channels))
        channelInoise = np.empty((len(beamCatalogPairIDs), args.channels))
        channelQnoise = np.empty((len(beamCatalogPairIDs), args.channels))
        channelUnoise = np.empty((len(beamCatalogPairIDs), args.channels))
        channelSourcePfracs = np.empty((len(beamCatalogPairIDs), args.channels))
        channelSourcePfracsErr = np.empty((len(beamCatalogPairIDs), args.channels))
    FWHMs = np.empty(args.channels) # full width half maximum, converted to degrees
    PBsigmas = np.empty(args.channels) # primary beam response sigma 

    for i in range(args.channels):
        # Format number to be a string literal of a 4 digit int, padded with zeros
        number = f"{i:04d}"

        # read in Q_i and copy its header, shape is (1,1,n,n)
        with fits.open(f"{args.pathQU}{number}-Q-image.fits") as hdulQ:
            dataQ = hdulQ[0].data[0,0]  # Assuming the data is in the primary HDU
            headerQ = hdulQ[0].header  
            freqQ = headerQ['CRVAL3'] # Hz

        # read in U_i
        with fits.open(f"{args.pathQU}{number}-U-image.fits") as hdulU:
            dataU = hdulU[0].data[0,0]  # Assuming the data is in the primary HDU

        # Ensure the two datasets have the same shape
        if dataQ.shape != dataU.shape:
            raise ValueError("The Q and U FITS files must have the same dimensions.")
        
        # Append channel frequency, calculate channel wavelength, FWHM, and primary beam response sigma
        channelFreqs[i] = freqQ
        channelWavelengths[i] = c/freqQ

        if args.correctForPBR:
            FWHMs[i] = np.degrees(1.09 * channelWavelengths[i] / args.antennaDiameter)
            PBsigmas[i] = FWHMs[i] / 2.35482004503

            # Produce a primary beam response image
            pbresponse = np.exp(-(x**2 + y**2) / (2 * (PBsigmas[i]/degPerPix)**2))
            pbresponse /= pbresponse.max()

        # Compute stokes P square root of the sum of the squares and append to the array
        Qnoise = np.nanstd(dataQ[emptyRegionRange])
        Unoise = np.nanstd(dataU[emptyRegionRange])
        channelP = np.sqrt(dataQ**2 + dataU**2) - Qnoise/2 - Unoise.std()/2
        
        if args.correctForPBR:
            channelPs[i] = channelP / pbresponse
        else:
            channelPs[i] = channelP

        # read in channel Stokes I image 
        with fits.open(f'{args.pathI}{number}-image.fits') as hdulChannelI: 
            dataChannelI = hdulChannelI[0].data[0,0]
        if args.correctForPBR:
            dataChannelI /= pbresponse # correct for primary beam response

        if args.doDiagnostics:
            # calculate Stokes I, Stokes P, polarization fraction, polarization angle for each source for each image
            Inoise = np.nanstd(dataChannelI[emptyRegionRange])
            for j, sourceID in enumerate(beamCatalogPairIDs):
                maxIndex = np.argmax(channelP[labeled_array == sourceID])
                sourceImax = dataChannelI[labeled_array == sourceID][maxIndex]
                sourcePmax = channelP[labeled_array == sourceID][maxIndex]
                sourceUmax = dataU[labeled_array == sourceID][maxIndex]
                sourceQmax = dataQ[labeled_array == sourceID][maxIndex]

                Perr = np.sqrt((sourceQmax**2 * Qnoise**2 + sourceUmax**2 * Unoise**2) / sourcePmax)
                sourcePfracErr = np.sqrt((Perr/sourceImax)**2 + (sourcePmax*Inoise/sourceImax**2)**2)

                channelSourceIs[j, i] = sourceImax
                channelSourcePs[j, i] = sourcePmax
                channelSourceQs[j, i] = sourceQmax
                channelSourceUs[j, i] = sourceUmax
                channelQnoise[j,i] = Qnoise
                channelUnoise[j,i] = Unoise
                channelSourcePfracs[j, i] = (channelP/dataChannelI)[labeled_array == sourceID][maxIndex] 
                channelSourcePfracsErr[j, i] = sourcePfracErr
                channelPolAngles[j, i] = np.arctan2(sourceUmax, sourceQmax) / 2

        # Save each P_i/I image to check leakage
        headerQ['CRVAL1'] += 360 # ra was negative, this fixes it to 0-360 range
        headerQ['CRVAL4'] = 4 # set the Stokes parameter to 4 (P)
        if args.saveChannelP:
            hdu = fits.PrimaryHDU(channelP/dataChannelI, header=headerQ) 
            hdu.writeto(f'{args.saveto}{number}-Pfrac-image.fits', overwrite=True)
            print(f'Successfully calculated P and saved P/I to `{args.saveto}{number}-Pfrac-image.fits` for channel {i}.')
        else:
            print(f'Successfully calculated P for channel {i}.')

    # stack P_i into an array of shape (channels,n,n) before averaging over channels
    # mask dataP values below SNR of PSNR
    dataP = np.nanmean(np.stack(channelPs, axis=0), axis=0)
    Pmask = np.where(dataP < args.PSNR * dataP[emptyRegionRange].std(), np.nan, 1)
    dataPmasked = dataP * Pmask

    if args.doWrite:
        # Save polarization to a new FITS file with mean channel frequency and Stokes 4
        headerQ['CRVAL3'] = np.mean(channelFreqs)
        hdu = fits.PrimaryHDU(dataP, header=headerQ) 
        hdu.writeto(f'{args.saveto}MFS-P-image.fits', overwrite=True)
        print(f"P image saved to `{args.saveto}MFS-P-image.fits`")

        # Save polarization fraction to a new FITS file with mean channel frequency and Stokes 4
        fracP = dataPmasked/dataImasked
        hdu = fits.PrimaryHDU(fracP, header=headerQ) 
        hdu.writeto(f'{args.saveto}MFS-Pfrac-image.fits', overwrite=True)
        print(f"P/I image saved to `{args.saveto}MFS-Pfrac-image.fits`")

        
    if args.doDiagnostics:
        save_plots(args, dataI, dataP, fracP, labeled_array, num_clusters,
                   channelFreqs, channelSourceIs, channelSourcePs, channelSourcePfracs,
                   channelSourcePfracsErr, channelSourceQs, channelSourceUs,
                   channelQnoise, channelUnoise, channelWavelengths, polCat,
                   catSourceXpixs, catSourceYpixs, catSourceNames, catSourceFreqs,
                   catSourceIs, catSourcePs, catSourcePolFracs, emptyRegionRange, beamCatalogPairIDs, channelPolAngles, polCatMasked, obsPBR if args.correctForPBR else None)
        




def save_plots(args, dataI, dataP, fracP, labeled_array, num_clusters, 
               channelFreqs, channelSourceIs, channelSourcePs, channelSourcePfracs,
               channelSourcePfracsErr, channelSourceQs, channelSourceUs,
               channelQnoise, channelUnoise, channelWavelengths, polCat,
               catSourceXpixs, catSourceYpixs, catSourceNames, catSourceFreqs,
               catSourceIs, catSourcePs, catSourcePolFracs, emptyRegionRange, beamCatalogPairIDs, channelPolAngles, polCatMasked, pbresponse=None):
    """Save diagnostic plots as PNG files."""
    
    # Verify empty region is source-free
    vmin, vmax = ZScaleInterval().get_limits(dataI)
    plt.figure(figsize=(8, 6), facecolor='gray')
    plt.imshow(dataI[emptyRegionRange], origin='lower', vmin=vmin, vmax=vmax)
    plt.title(f'Empty region for observation SB{args.SBID}')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{args.saveto}empty_region.png', bbox_inches='tight')
    plt.close()

    # Plot masked polarization fraction
    print(f'Decimal fraction of non nan values {np.count_nonzero(~np.isnan(fracP)) / fracP.size:.2g}, ({np.count_nonzero(~np.isnan(fracP)) / fracP.size*100:.2g}%)')

    plt.figure(figsize=(12, 9), facecolor='gray')
    plt.imshow(fracP, origin='lower', vmin=None, vmax=None, cmap='viridis')

    for i in range(1, len(catSourceXpixs)+1):
        plt.scatter(catSourceXpixs[i-1], catSourceYpixs[i-1], color='blue', s=30, facecolors='blue', label=f'Catalog Source {i}')
        plt.text(catSourceXpixs[i-1], catSourceYpixs[i-1]-50, f'{i}', color='blue', fontsize=8, ha='center', va='center')
    
    plt.title(f'Bright sources with polarization fraction, masked out pixels of P < {args.PSNR}σₚ and I < {args.ISNR}σᵢ')
    plt.legend()
    plt.xticks([])
    plt.yticks([])

    # Label each cluster with polarization fraction
    for cluster_id in range(1, num_clusters + 1):
        cluster_values = fracP[labeled_array == cluster_id] 
        max_value = np.nanmax(cluster_values)
        yC, xC = np.argwhere(labeled_array == cluster_id)[0]
        plt.text(xC, yC+70, f'{cluster_id}: {max_value*100:.1f}%', color='red', fontsize=8, ha='center', va='center')

    # Circle polarized pixels
    yP, xP = np.where(~np.isnan(fracP))
    plt.scatter(xP, yP, color='red', s=100, linewidth=1, facecolors='none', label='Non-NaN Pixels')
    plt.savefig(f'{args.saveto}polarization_fraction.png', bbox_inches='tight')
    plt.close()

    # Plot P flux density
    vmin, vmax = ZScaleInterval().get_limits(dataP)
    plt.figure(figsize=(12, 9), facecolor='gray')
    plt.imshow(dataP, origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
    plt.title('Polarization flux density')
    plt.colorbar(label='Jy/beam')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{args.saveto}polarization_flux.png', bbox_inches='tight')
    plt.close()

    # Plot primary beam response if corrected
    if args.correctForPBR and pbresponse is not None:
        plt.figure(figsize=(12, 9))
        plt.imshow(pbresponse, origin='lower')
        for xCS,yCS,label in zip(catSourceXpixs[:-1], catSourceYpixs[:-1], beamCatalogPairIDs):
            plt.scatter(xCS,yCS, label=f'Source {label} with pbr {pbresponse[round(xCS),round(yCS)]}')
        plt.legend()
        plt.xticks([]) 
        plt.yticks([])
        plt.title(f'Primary beam response at {round(channelFreqs[-1]/1e6)} MHz')
        plt.savefig(f'{args.saveto}primary_beam_response.png', bbox_inches='tight')
        plt.close()

    # Plot Stokes I flux density
    plt.figure(figsize=(12, 9), facecolor='gray')
    vmin, vmax = ZScaleInterval().get_limits(dataI)
    plt.imshow(dataI, origin='lower', vmin=vmin, vmax=vmax)
    plt.title('Stokes I flux intensity')
    plt.colorbar(label='Jy/beam')
    plt.xticks([]) 
    plt.yticks([])
    plt.savefig(f'{args.saveto}stokes_I_flux.png', bbox_inches='tight')
    plt.close()

    # Plot Stokes I over frequency
    plt.figure(figsize=(10, 6))
    for i in range(0, len(channelSourceIs)):
        plt.plot(channelFreqs/1e9, channelSourceIs[i], label=f'Source {i+1}')

    plt.scatter(catSourceFreqs[0]/1e9, catSourceIs[0], label='Catalog Source 1', color='blue')
    plt.scatter(catSourceFreqs[1]/1e9, catSourceIs[1], label='Catalog Source 2', color='orange')
    plt.scatter(catSourceFreqs[2]/1e9, catSourceIs[2], label='Catalog Source 2', marker='*', color='orange')
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Stokes I (Jy/beam)')
    plt.savefig(f'{args.saveto}stokesI_vs_freq.png', bbox_inches='tight')
    plt.close()

    # Plot P/I over frequency
    plt.figure(figsize=(10, 6))
    for i in range(len(channelSourcePfracs)):
        plt.errorbar(channelFreqs/1e9, channelSourcePfracs[i], channelSourcePfracsErr[i], 
                    label=catSourceNames[i], markersize=4, capsize=5, ecolor='black')

    plt.scatter(catSourceFreqs[0]/1e9, catSourcePolFracs[0], label='J1911+1923', color='blue')
    plt.scatter(catSourceFreqs[1]/1e9, catSourcePolFracs[1], label='J1913+2019', color='orange')
    plt.scatter(catSourceFreqs[2]/1e9, catSourcePolFracs[2], label='J1913+2019', marker='*', s=50, color='orange')
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Polarization Fraction')
    plt.savefig(f'{args.saveto}polfrac_vs_freq.png', bbox_inches='tight')
    plt.close()

    # Plot polarization intensity vs frequency
    plt.figure(figsize=(10, 6))
    for i in range(0, len(channelSourcePs)):
        plt.plot(channelFreqs/1e9, channelSourcePs[i], label=f'Source {i+1}')

    plt.scatter(catSourceFreqs[0]/1e9, catSourcePs[0], label='Catalog Source 1', color='blue')
    plt.scatter(catSourceFreqs[1]/1e9, catSourcePs[1], label='Catalog Source 2', color='orange')
    plt.scatter(catSourceFreqs[2]/1e9, catSourcePs[2], label='Catalog Source 2', marker='*', color='orange')
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Polarization Intensity (Jy/beam)')
    plt.savefig(f'{args.saveto}polint_vs_freq.png', bbox_inches='tight')
    plt.close()

    # Plot polarization angle vs wavelength and fit RM
    channelPolAnglesE = unumpy.arctan2(unumpy.uarray(channelSourceUs, channelUnoise), 
                                      unumpy.uarray(channelSourceQs, channelUnoise)) / 2
    linearizedChannelPolAngles = np.unwrap(unumpy.nominal_values(channelPolAnglesE), period=np.pi)

    coefficients, polAngleCovMatrix = np.polyfit(channelWavelengths**2, linearizedChannelPolAngles.T, 1, cov=True)
    RM, b = coefficients

    colors = ['purple', 'green']
    plt.figure(figsize=(10, 6))
    for i in range(0, len(channelPolAngles)):
        [RMerr, berr] = np.sqrt(np.diag(polAngleCovMatrix[i]))
        print(f'Uncertainty in RM parameter {RMerr}')
        plt.errorbar(channelWavelengths, linearizedChannelPolAngles[i], 
                    unumpy.std_devs(channelPolAnglesE[i]), ecolor='black', 
                    label=catSourceNames[i], fmt='o', capsize=3, capthick=1, markersize=3)
        plt.plot(channelWavelengths, RM[i]*channelWavelengths**2 + b[i], 
                label=f'fitted RM = {RM[i]:.0f} ± {1} (rad/m²)', zorder=20, 
                color=colors[i%len(colors)], lw=4, alpha=.3)
    plt.xlabel(r'Wavelength (m)')
    plt.ylabel('Polarization Angle (rad)')
    plt.legend()
    plt.savefig(f'{args.saveto}polangle_vs_wavelength.png', bbox_inches='tight')
    plt.close()

    print(polCatMasked['rm'])

    # Plot RM distribution from catalog
    rm_data = polCat['rm']
    lower_bound = np.percentile(rm_data, 1)
    upper_bound = np.percentile(rm_data, 99)
    filtered_rm = rm_data[(rm_data >= lower_bound) & (rm_data <= upper_bound)]

    plt.figure(figsize=(10, 6))
    plt.hist(filtered_rm, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Faraday Depth (rad m$^{-2}$)')
    plt.ylabel('Count')
    plt.title('Histogram of Faraday Depth')
    plt.grid(True)
    plt.savefig(f'{args.saveto}rm_distribution.png', bbox_inches='tight')
    plt.close()

    # Plot histogram of polarization fractions
    plt.figure(figsize=(10, 6))
    plt.hist(fracP.flatten(), bins=50)
    plt.title('Polarization Fraction histogram')
    plt.ylabel('counts')
    plt.xlabel('Polarization fraction')
    plt.savefig(f'{args.saveto}polfrac_histogram.png', bbox_inches='tight')
    plt.close()



def main():
    """Execute the script."""
    args = parse_arguments()
    calculate_polarization(args)

if __name__ == "__main__":
    main()