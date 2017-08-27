import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import os.path
    import sys

    path2therpy = os.path.join(os.path.expanduser('~'), 'Documents', 'My Programs', 'Python Library')
    sys.path.append(path2therpy)
    from therpy import smooth
    from therpy import calculus
    from therpy import imageio
    from therpy import imagedata
    from therpy import constants
    from therpy import functions
else:
    from . import smooth
    from . import calculus
    from . import imageio
    from . import imagedata
    from . import constants
    from . import functions
    from . import classes


################################################

class hybridEoS_avg:
    '''
    kwargs -- required
        cst = tp.cst(sigmaf=0.5, Nsat=200, pixel=0.68e-6, trapw=2*np.pi*23.9)
        cropi or [center=(x,y), width=400, height=400]
    kwargs -- optional
        bgwidth = 20
        odlim or [odlimA = (0,2), odlimB = (0,2)]
        bgplot = False
        Xyuse or [XyuseA = None, XyuseB = None]  this will be selected by fitting TF profile and yset
        Xmethod or [XmethodA = 'linear', XmethodB = 'linear']
        Xplot or [XplotA = False, XplotB = False]
        Xyset or [XysetA = (10, 0.75), XysetB = (10, 0.75)]  fit circles to 75% of the center and 10 pixels average
        Xguess or [XguessA = None, XguessB = None]

        usecenter = 'A', 'B', or None, default is None
        Ncor or [NcorA = 1, NcorB = 1]
        nmax or [nmaxA = np.inf, nmaxB = np.inf]


    '''
    def __init__(self, names, **kwargs):
        self.var = kwargs
        self.processNames(names)
        self.computeOD()

    @property
    def cst(self):
        return self.var['cst']

    @property
    def cropi(self):
        return self.var['cropi']

    @property
    def odfixs(self):
        if 'odfixA' not in self.var.keys():
            self.fixOds()
        return (self.var['odfixA'], self.var['odfixB'])

    @property
    def xsecs(self):
        if 'xsecA' not in self.var.keys():
            self.getXsecs()
        return (self.var['xsecA'], self.var['xsecB'])

    @property
    def odA(self):
        return self.xsecs[0].od

    @property
    def odB(self):
        return self.xsecs[1].od

    @property
    def nzA(self):
        if 'nzA' not in self.var.keys():
            self.OD2Densities()
        return self.var['nzA']

    @property
    def nzB(self):
        if 'nzB' not in self.var.keys():
            self.OD2Densities()
        return self.var['nzB']

    @property
    def nuA(self):
        u = 0.5 * self.cst.mass * self.cst.trapw ** 2 * self.nzA.x ** 2
        return classes.Curve(x=u, y=self.nzA.y, xscale=self.cst.h * 1e3, yscale=1).sortbyx()

    @property
    def nuB(self):
        u = 0.5 * self.cst.mass * self.cst.trapw ** 2 * self.nzB.x ** 2
        return classes.Curve(x=u, y=self.nzB.y, xscale=self.cst.h * 1e3, yscale=1).sortbyx()

    @property
    def EFuA(self):
        u = 0.5 * self.cst.mass * self.cst.trapw ** 2 * self.nzA.x ** 2
        return classes.Curve(x=u, y=self.cst.n2EF(self.nzA.y, True),
                             xscale=self.cst.h*1e3, yscale=self.cst.h*1e3).sortbyx()

    @property
    def EFuB(self):
        u = 0.5 * self.cst.mass * self.cst.trapw ** 2 * self.nzB.x ** 2
        return classes.Curve(x=u, y=self.cst.n2EF(self.nzB.y, True),
                             xscale=self.cst.h*1e3, yscale=self.cst.h*1e3).sortbyx()

    def processNames(self, names):
        # The format of input 'names'
        if type(names) is str:
            namesA = [names]
            namesB = [names[0:-1] + 'B']
        elif type(names) is list:
            namesA = names[0::2]
            namesB = names[1::2]
        elif type(names).__module__ == 'pandas.core.series':
            names_ = names.tolist()
            namesA = names_[0::2]
            namesB = names_[1::2]
        else:
            raise ValueError("Invalid input for names")
        self.var['namesA'] = namesA
        self.var['namesB'] = namesB

    def computeOD(self):
        # Get the first OD
        namesA = self.var['namesA']
        namesB = self.var['namesB']
        imA = classes.AbsImage(name=namesA[0], cst=self.cst)
        imB = classes.AbsImage(name=namesB[0], cst=self.cst)
        odA = imA.od
        odB = imB.od
        # Average rest of the images
        for i, nA in enumerate(namesA[1:]):
            odA += classes.AbsImage(name=namesA[i], cst=self.cst).od
            odB += classes.AbsImage(name=namesB[i], cst=self.cst).od
        odA /= len(namesA)
        odB /= len(namesB)
        # Get cropi
        if 'cropi' not in self.var.keys():
            self.var['cropi'] = imA.cropi(**self.var)
        # Store
        self.var['odA'] = odA
        self.var['odB'] = odB

    def fixOds(self):
        if 'odlim' in self.var.keys():
            self.var['odlimA'], self.var['odlimB'] = self.var['odlim'], self.var['odlim']
        odfixA = classes.ODFix2D(od=self.var['odA'],
                                 cropi=self.cropi,
                                 width=self.var.get('bgwidth', 20),
                                 odlim=self.var.get('odlimA', (0, 2)),
                                 plot=self.var.get('bgplot', False))
        odfixB = classes.ODFix2D(od=self.var['odB'],
                                 cropi=self.cropi,
                                 width=self.var.get('bgwidth', 20),
                                 odlim=self.var.get('odlimB', (0, 2)),
                                 plot=self.var.get('bgplot', False))
        self.var['odfixA'] = odfixA
        self.var['odfixB'] = odfixB

    def getXsecs(self):
        # process inputs
        if 'Xyuse' in self.var.keys():
            self.var['XuseA'], self.var['XuseB'] = self.var['Xyuse'], self.var['Xyuse']
        if 'Xmethod' in self.var.keys():
            self.var['XmethodA'], self.var['XmethodB'] = self.var['Xmethod'], self.var['Xmethod']
        if 'Xplot' in self.var.keys():
            self.var['XplotA'], self.var['XplotB'] = self.var['Xplot'], self.var['Xplot']
        if 'odlim' in self.var.keys():
            self.var['odlimA'], self.var['odlimB'] = self.var['odlim'], self.var['odlim']
        if 'Xyset' in self.var.keys():
            self.var['XysetA'], self.var['XysetB'] = self.var['Xyset'], self.var['Xyset']
        if 'Xguess' in self.var.keys():
            self.var['XguessA'], self.var['XguessB'] = self.var['Xguess'], self.var['Xguess']

        # calculate cross sections
        self.var['xsecA'] = classes.XSectionTop(od=self.odfixs[0].od,
                                                yuse=self.var.get('XyuseA', None),
                                                method=self.var.get('XmethodA', 'linear'),
                                                plot=self.var.get('XplotA', False),
                                                odlim=self.var.get('odlimA', (0, 2)),
                                                yset=self.var.get('XysetA', (10, 0.75)),
                                                guess=self.var.get('XguessA', None))
        self.var['xsecB'] = classes.XSectionTop(od=self.odfixs[1].od,
                                                yuse=self.var.get('XyuseB', None),
                                                method=self.var.get('XmethodB', 'linear'),
                                                plot=self.var.get('XplotB', False),
                                                odlim=self.var.get('odlimB', (0, 2)),
                                                yset=self.var.get('XysetB', (10, 0.75)),
                                                guess=self.var.get('XguessB', None))

    def OD2Densities(self):
        # Process inputs
        if 'Ncor' in self.var.keys():
            NcorA = self.var['Ncor']
            NcorB = NcorA
        else:
            NcorA = self.var.get('NcorA', 1)
            NcorB = self.var.get('NcorB', 1)
        if 'nmax' in self.var.keys():
            self.var['nmaxA'], self.var['nmaxB'] = self.var['nmax'], self.var['nmax']

        if self.var.get('usecenter', None) == 'A':
            self.var['od2denA'] = classes.OD2Density(od=self.odA, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxA', np.inf),
                                                     Ncor=NcorA, plot=False, center=None)
            self.var['od2denB'] = classes.OD2Density(od=self.odB, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxB', np.inf),
                                                     Ncor=NcorB, plot=False, center=self.var['od2denA'].center)
        elif self.var.get('usecenter', None) == 'B':
            self.var['od2denB'] = classes.OD2Density(od=self.odB, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxB', np.inf),
                                                     Ncor=NcorB, plot=False, center=None)
            self.var['od2denA'] = classes.OD2Density(od=self.odA, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxA', np.inf),
                                                     Ncor=NcorA, plot=False, center=self.var['od2denB'].center)
        else:
            self.var['od2denA'] = classes.OD2Density(od=self.odA, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxA', np.inf),
                                                     Ncor=NcorA, plot=False, center=None)
            self.var['od2denB'] = classes.OD2Density(od=self.odB, xsec=self.xsecs[0], pixel=self.cst.pixel,
                                                     sigma=self.cst.sigma, nmax=self.var.get('nmaxB', np.inf),
                                                     Ncor=NcorB, plot=False, center=None)

        self.var['nzA'] = self.var['od2denA'].nz
        self.var['nzB'] = self.var['od2denB'].nz

    def infoplot(self, odlimA=None, odlimB=None):
        if 'odlim' in self.var.keys():
            self.var['odlimA'], self.var['odlimB'] = self.var['odlim'], self.var['odlim']
        if odlimA is None: odlimA = self.var.get('odlimA', (0, 2))
        if odlimB is None: odlimB = self.var.get('odlimB', (0, 2))

        fig = plt.figure(figsize=(15, 4))
        ax = plt.subplot2grid((4, 3), (0, 0), rowspan=4)
        axt = plt.subplot2grid((4, 3), (0, 1), rowspan=4)
        ax1 = plt.subplot2grid((4, 3), (0, 2))
        ax2 = plt.subplot2grid((4, 3), (1, 2))
        ax3 = plt.subplot2grid((4, 3), (2, 2), rowspan=2)

        ax.scatter(self.nzA.x * 1e6, self.nzA.y, color='blue', alpha=0.4, marker='.', s=7)
        ax.plot(self.nzA.xyfitplot[0] * 1e6, self.nzA.xyfitplot[1], 'k-')
        ax.scatter(self.nzB.x * 1e6, self.nzB.y, color='red', alpha=0.4, marker='.', s=7)
        ax.plot(self.nzB.xyfitplot[0] * 1e6, self.nzB.xyfitplot[1], 'k-')
        ax.plot([0, 0], [min(self.nzA.miny, self.nzB.miny), 1.1 * max(self.nzA.maxy, self.nzB.maxy)], 'k--')
        ax.set(xlabel='z [$\mu$m]', ylabel='n [m$^{-3}$]')

        axt.plot(*self.EFuA.plotdata, 'b-', alpha=0.6)
        axt.plot(*self.EFuB.plotdata, 'r-', alpha=0.6)
        axt.plot([0, 1.1 * np.max(self.EFuA.plotdata[0])], [0, 0], 'k--')
        axt.set( xlabel='U [kHz]', ylabel='$E_F$ [kHz]')

        ax1.scatter(self.xsecs[0].ycen_, self.xsecs[0].xl_, s=1, color='white')
        ax1.scatter(self.xsecs[0].ycen_, self.xsecs[0].xr_, s=1, color='white')
        ax1.scatter(self.xsecs[0].ycen_, self.xsecs[0].xc_, s=1, color='k')
        ax1.imshow(self.xsecs[0].od.T, clim=odlimA, aspect='auto', cmap='viridis', origin='lower')
        ax1.set(xlim=[self.xsecs[0].yall[0], self.xsecs[0].yall[-1]], title='OD and radius')
        ax1.set_axis_off()

        ax2.scatter(self.xsecs[1].ycen_, self.xsecs[1].xl_, s=1, color='white')
        ax2.scatter(self.xsecs[1].ycen_, self.xsecs[1].xr_, s=1, color='white')
        ax2.scatter(self.xsecs[1].ycen_, self.xsecs[1].xc_, s=1, color='k')
        ax2.imshow(self.xsecs[1].od.T, clim=odlimB, aspect='auto', cmap='viridis', origin='lower')
        ax2.set(xlim=[self.xsecs[1].yall[0], self.xsecs[1].yall[-1]])
        ax2.set_axis_off()

        ax3.plot(self.xsecs[0].yall, self.xsecs[0].rad, 'k-')
        ax3.scatter(self.xsecs[0].ycen_, self.xsecs[0].r_, color='blue')
        ax3.set(xlim=[self.xsecs[0].yall[0], self.xsecs[0].yall[-1]])
        ax3.plot(self.xsecs[1].yall, self.xsecs[1].rad, 'k-')
        ax3.scatter(self.xsecs[1].ycen_, self.xsecs[1].r_, color='red')
        ax3.set(xlim=[self.xsecs[1].yall[0], self.xsecs[1].yall[-1]])
        ax3.set(xlabel='z [pixel]', ylabel='radius [pixels]')

    def getThermoPlot(self,**sett):
        # Bin EFu
        EFuA = self.EFuA.binbyx(bins = sett.get('binsA', (20,10)),
                                edges = sett.get('edgesA', self.EFuA.maxx/3))
        EFuB = self.EFuB.binbyx(bins = sett.get('binsB', (20,10)),
                                edges = sett.get('edgesB', self.EFuB.maxx/3))
        # Get kappa
        kuA = EFuA.diff(method='poly',order=sett.get('dorder',1), points=sett.get('dpoints',1))
        kuA = classes.Curve(kuA.x, -kuA.y, xscale=self.cst.h*1e3, yscale=1)
        kuB = EFuB.diff(method='poly', order=sett.get('dorder', 1), points=sett.get('dpoints', 1))
        kuB = classes.Curve(kuB.x, -kuB.y, xscale=self.cst.h*1e3, yscale=1)

        # store
        self.thermo = dict(EFuA=EFuA, EFuB=EFuB, kuA = kuA, kuB = kuB)

        # Plot
        fig,ax = plt.subplots(ncols=2,figsize=(10,4))
        ax[0].plot(*self.EFuA.plotdata,'b-',alpha=0.3)
        ax[0].plot(*EFuA.plotdata,'b.')
        ax[0].plot(*self.EFuB.plotdata, 'r-', alpha=0.3)
        ax[0].plot(*EFuB.plotdata, 'r.')
        ax[0].plot([0, 1.1*EFuA.maxx/EFuA.xscale],[0,0],'k--')
        ax[0].set(xlabel='U [kHz]', ylabel='$E_F$ [kHz]')

        ax[1].plot(*kuA.plotdata, 'b-')
        ax[1].plot(*kuA.plotdata, 'b.')
        ax[1].plot(*kuB.plotdata, 'r-')
        ax[1].plot(*kuB.plotdata, 'r.')
        ax[1].set(xlabel='U [kHz]', ylabel='$\kappa / \kappa_0$')

# Thermodynamics Class
class ThermodynamicsHybrid:
    def __init__(self, nz, cst, **kwargs):
        self.var = kwargs
        self.var['nz_'] = nz
        self.cst = cst

    #def