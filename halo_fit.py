from headers import *

###################################################################

class ParamsHalofit(object):
   
   def __init__(self, U, z):
      self.U = U
      
      # fitting functions
      Omz = self.U.OmM * (1 + z)**3 / (self.U.OmM * (1 + z)**3 + (1.-self.U.OmM))
      self.f1 = Omz**(-0.0307)
      self.f2 = Omz**(-0.0585)
      self.f3 = Omz**(0.0743)

      # Find R_sigma such that sigma2(R_sigma) = 1
      W3d_g = lambda x: np.exp(-0.5 * x**2)
      f = lambda R, z: self.U.Sigma2(R, z, W3d_g) - 1.
      R_sigma = optimize.brentq(f, 0.001, 100., xtol=1e-3, args=z)
      self.k_sigma = 1. / R_sigma

      # derivatives of sigma2 at the non-linear scale
      h = 0.01
      dsdR = (self.U.Sigma2(R_sigma+h, z, W3d_g) - self.U.Sigma2(R_sigma-h, z, W3d_g))/(2 * h)
      d2sdR2 = (self.U.Sigma2(R_sigma+h, z, W3d_g) + self.U.Sigma2(R_sigma-h, z, W3d_g) - 2 * self.U.Sigma2(R_sigma, z, W3d_g))/(h**2)

      self.neff = - R_sigma * dsdR - 3.
      self.C = - R_sigma * dsdR + R_sigma**2 * dsdR**2 - R_sigma**2 * d2sdR2

      self.an = 10**( 1.5222 + 2.8553 * self.neff + 2.3706 * self.neff**2 + 0.9903 * self.neff**3 + 0.2250 * self.neff**4 - 0.6038 * self.C )
      self.bn = 10**( -0.5642 + 0.5864 * self.neff + 0.5716 * self.neff**2 - 1.5474 * self.C )
      self.cn = 10**( 0.3696 + 2.0404 * self.neff + 0.8161 * self.neff**2 + 0.5869 * self.C )
      self.gamman = 0.1971 - 0.0843 * self.neff + 0.8460 * self.C
      self.alphan = abs(6.0835 + 1.3373 * self.neff - 0.1959 * self.neff**2 - 5.5274 * self.C)
      self.betan = 2.0379 - 0.7354 * self.neff + 0.3157 * self.neff**2 + 1.2490 * self.neff**3 + 0.3980 * self.neff**4 - 0.1682 * self.C
      self.mun = 0.
      self.nun = 10**( 5.2105 + 3.6902 * self.neff )


###################################################################

class Halofit(object):
   """Non-Linear Matter Power spectrum
   Based on fits by Takahashi et al (2012)
   arXiv:1208.2701v2
   adapted from Simone Ferraro's code
   """
   
   def __init__(self, U, save=False):
      self.U = U

      # values of k to compute
      self.Nk = 50
      self.kmin = 1.e-3
      self.kmax = 1.e2
      self.kvec = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.Nk, 10.)

      # values of z to compute
      self.Nz = 40
      self.zmin = 0.
      self.zmax = 6.
      self.zvec = np.linspace(self.zmin, self.zmax, self.Nz)
   
      # create folder if needed
      directory = "./output/halofit/"
      if not os.path.exists(directory):
         os.makedirs(directory)

      # save if needed, then load
      self.path = "./output/halofit/halofit_Planck_Nk" + str(self.Nk) + "_Nz" + str(self.Nz) + "_zmin" + str(self.zmin) + "_zmax" + str(self.zmax) + ".txt"
      if (save==True) or (not os.path.exists(self.path)):
         self.SaveAll()
      self.LoadAll()


   def fPhalofit(self, k, z, A):
      """A has to be a ParamsHalofit object
      """
      #print '********** k, z = ', k, z
      Delta2Linz = k**3 * self.U.fPlin_z(k, z)/(2 * np.pi**2)
      
      y = k / A.k_sigma
      fy = y / 4. + y**2 / 8.
      Delta2Q = Delta2Linz * ( (1 + Delta2Linz)**A.betan / (1 + A.alphan * Delta2Linz) ) * np.exp(-fy)      # 2-halo term
      Delta2Hprime = A.an * y**(3 * A.f1) / (1 + A.bn * y**(A.f2) + (A.cn * A.f3 * y)**(3. - A.gamman))
      Delta2H = Delta2Hprime / (1. + A.mun / y + A.nun / y**2)    # 1-halo term
      Delta2NL = Delta2Q + Delta2H
      
      return (2 * np.pi**2) / k**3 * Delta2NL


   def SaveAll(self):
      Pmat = np.zeros((self.Nk, self.Nz))
      for iZ in range(self.Nz):
         z = self.zvec[iZ]
         A = ParamsHalofit(self.U, z)
         for iK in range(self.Nk):
            k = self.kvec[iK]
            Pmat[iK, iZ] = self.fPhalofit(k, z, A)
         print('done with redshift',z)
      np.savetxt(self.path, Pmat)


   def LoadAll(self):
      self.Pmat = np.genfromtxt(self.path)
      f = RectBivariateSpline(np.log(self.kvec), self.zvec, np.log(self.Pmat), s=0)
      self.fPinterp = lambda k, z: (k>=self.kmin)*(k<=self.kmax)*(z>=self.zmin)*(z<=self.zmax) * np.exp(f(np.log(k), z))
      self.fP_2h = self.fPinterp
      self.fP2hinterp = self.fPinterp
      self.fP_1h = lambda k, z: 0.
      self.fP1hinterp = lambda k, z: 0.


   def plotP(self):

      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      for iZ in range(0, self.Nz, 3):
      #for iZ in [self.Nz-1]:
         z = self.zvec[iZ]
         ax.loglog(self.kvec, self.Pmat[:,iZ], 'b-', label=r'$z=$'+str(z))
         #
         Plin = np.array([self.U.fPlin_z(k, z) for k in self.kvec])
         ax.loglog(self.kvec, Plin, 'k--')
      #
      ax.legend(loc=1)
      ax.set_xlabel(r'$k$ [h/Mpc]')
      ax.set_ylabel(r'$P(k)$ [(Mpc/h)$^3$]')
      
      plt.show()





