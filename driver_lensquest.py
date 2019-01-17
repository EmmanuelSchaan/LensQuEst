import universe
reload(universe)
from universe import *

import halo_fit
reload(halo_fit)
from halo_fit import *

import weight
reload(weight)
from weight import *

import pn_2d
reload(pn_2d)
from pn_2d import *

import cmb
reload(cmb)
from cmb import *

import flat_map
reload(flat_map)
from flat_map import *


##################################################################################
##################################################################################
print "Map properties"

# number of pixels for the flat map
nX = 400 #1200
nY = 400 #1200

# map dimensions in degrees
sizeX = 10.
sizeY = 10.

# basic map object
baseMap = FlatMap(nX=nX, nY=nY, sizeX=sizeX*np.pi/180., sizeY=sizeY*np.pi/180.)

# multipoles to include in the lensing reconstruction
lMin = 30.; lMax = 3.5e3

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra


##################################################################################
print "CMB experiment properties"

# Adjust the lMin and lMax to the assumptions of the analysis
# CMB S3 specs
cmb = StageIVCMB(beam=1., noise=1., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)

# Total power spectrum, for the lens reconstruction
forCtotal = lambda l: cmb.flensedTT(l) + cmb.fdetectorNoise(l)
#
# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(map(forCtotal, L))
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)


##################################################################################
print "CMB lensing power spectrum"

u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)


##################################################################################
##################################################################################
print "Compute the statistical uncertainty on the reconstructed lensing convergence"

print "- standard quadratic estimator"
fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
print "- magnification estimator"
fNdCmb_fft = baseMap.forecastN0KappaDilation(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)
print "- shear E-mode estimator"
fNsCmb_fft = baseMap.forecastN0KappaShear(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)
#print "- shear B-mode estimator"
#fNsBCmb_fft = baseMap.forecastN0KappaShearB(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)
#print "- shear E x magnification"
#fNsdCmb_fft = baseMap.forecastN0KappaShearDilation(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMaxS=lMax, lMaxD=lMax, corr=True, test=False)
#print "- shear E x shear B. Not yet working."
#fNssBCmb_fft = baseMap.forecastN0KappaShearShearB(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, corr=True, test=False)


##################################################################################
# colors for plots

cQ = 'r'
cS = 'b'
cD = 'g'


##################################################################################
print "Plot noise power spectra"

fig=plt.figure(0)
ax=fig.add_subplot(111)
#
ax.loglog(L, p2d_cmblens.fPinterp(L), 'k-', lw=3, label=r'signal')
#
Nq = fNqCmb_fft(L)
Ns = fNsCmb_fft(L)
Nd = fNdCmb_fft(L)
#Nsd = fNsdCmb_fft(L)
#NsB = fNsBCmb_fft(L)
#NssB = fNssBCmb_fft(L)
#
ax.loglog(L, Ns, c=cS, label=r'shear')
ax.loglog(L, Nd, c=cD, label=r'dilation')
ax.loglog(L, Nq, c=cQ, lw=3, label=r'QE')
#ax.loglog(L, NsB, c='y', label=r'shear B')a
#x.loglog(L, Nsd, c='c', ls='--', label=r'shear$\times$dilation')
#ax.loglog(L, 1./(1./Nd + 1./Ns), c=cQ, ls='-.', label=r'naive shear + dilation')
#ax.loglog(L, (Ns*Nd-Nsd**2)/(Ns+Nd-2.*Nsd), c=cQ, ls='--', lw=3, label=r'shear + dilation')
#ax.loglog(L, NssB, c='y', ls='-.', label=r'shear$\times$shear B')
#
ax.legend(loc=2, fontsize='x-small', labelspacing=0.1)
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'$C_L^\kappa$')

plt.show()


##################################################################################
print "Generate GRF unlensed CMB map (debeamed)"

cmb0Fourier = baseMap.genGRF(cmb.funlensedTT, test=False)
cmb0 = baseMap.inverseFourier(cmb0Fourier)
print "plot unlensed CMB map"
baseMap.plot(cmb0)
print "check the power spectrum"
lCen, Cl, sCl = baseMap.powerSpectrum(cmb0Fourier, theory=[cmb.funlensedTT], plot=True, save=False)


##################################################################################
print "Generate GRF kappa map"

kCmbFourier = baseMap.genGRF(p2d_cmblens.fPinterp, test=False)
kCmb = baseMap.inverseFourier(kCmbFourier)
print "plot kappa map"
baseMap.plot(kCmb)
print "check the power spectrum"
lCen, Cl, sCl = baseMap.powerSpectrum(kCmbFourier, theory=[p2d_cmblens.fPinterp], plot=True, save=False)


##################################################################################
print "Lens the CMB map"

lensedCmb = baseMap.doLensing(cmb0, kappaFourier=kCmbFourier)
lensedCmbFourier = baseMap.fourier(lensedCmb)
print "plot lensed CMB map"
baseMap.plot(lensedCmb, save=False)
print "check the power spectrum"
lCen, Cl, sCl = baseMap.powerSpectrum(lensedCmbFourier, theory=[cmb.funlensedTT, cmb.flensedTT], plot=True, save=False)


##################################################################################
print "Add white detector noise (debeamed)"

noiseFourier = baseMap.genGRF(cmb.fdetectorNoise, test=False)
totalCmbFourier = lensedCmbFourier + noiseFourier
totalCmb = baseMap.inverseFourier(totalCmbFourier)
baseMap.plot(totalCmb)
print "check the power spectrum"
lCen, Cl, sCl = baseMap.powerSpectrum(totalCmbFourier,theory=[cmb.funlensedTT, cmb.flensedTT, cmb.fCtotal], plot=True, save=False)


##################################################################################
print "Reconstructing kappa: standard quadratic estimator"

pathQCmb = "./output/qCmb.txt"
baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalCmbFourier, test=False, path=pathQCmb)
qCmbFourier = baseMap.loadDataFourier(pathQCmb)

print "Auto-power: kappa_rec"
lCen, Cl, sCl = baseMap.powerSpectrum(qCmbFourier,theory=[p2d_cmblens.fPinterp, fNqCmb_fft], plot=True, save=False)

print "Cross-power: kappa_rec x kappa_true"
lCen, Cl, sCl = baseMap.crossPowerSpectrum(qCmbFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp, fNqCmb_fft], plot=True, save=False)


##################################################################################
print "Reconstructing kappa: shear estimator"

pathSCmb = "./output/sCmb.txt"
baseMap.computeQuadEstKappaShearNormCorr(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalCmbFourier, test=False, path=pathSCmb)
sCmbFourier = baseMap.loadDataFourier(pathSCmb)

print "Auto-power: kappa_rec"
lCen, Cl, sCl = baseMap.powerSpectrum(sCmbFourier,theory=[p2d_cmblens.fPinterp, fNsCmb_fft], plot=True, save=False)

print "Cross-power: kappa_rec x kappa_true"
lCen, Cl, sCl = baseMap.crossPowerSpectrum(sCmbFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp, fNsCmb_fft], plot=True, save=False)


##################################################################################
print "Reconstructing kappa: magnification estimator"

pathDCmb = "./output/dCmb.txt"
baseMap.computeQuadEstKappaDilationNormCorr(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=totalCmbFourier, test=False, path=pathDCmb)
dCmbFourier = baseMap.loadDataFourier(pathDCmb)

print "Auto-power: kappa_rec"
lCen, Cl, sCl = baseMap.powerSpectrum(dCmbFourier,theory=[p2d_cmblens.fPinterp, fNdCmb_fft], plot=True, save=False)

print "Cross-power: kappa_rec x kappa_true"
lCen, Cl, sCl = baseMap.crossPowerSpectrum(dCmbFourier, kCmbFourier, theory=[p2d_cmblens.fPinterp, fNdCmb_fft], plot=True, save=False)

