import numpy as np
from matplotlib import pyplot as plt
from pylab import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process.kernels import Matern
import pandas as pd
import scipy.optimize as opt
import george
from george.kernels import ExpSquaredKernel, Matern32Kernel
#from astrophysics import phot
#from pyastrolib

rc('text', usetex=True)
rc('font',family='Times New Roman')
rc('xtick', labelsize=13)
rc('ytick', labelsize=13)

np.random.seed(1)

# ----------------------------------------------------------------------
#  First the noiseless case
#X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Observations
#y = f(X).ravel()

lcf = open('/Users/uramito/Dropbox/DES_paper/ptf12dam_gr.txt','r')      # def di file-oggetto
riga = lcf.readlines()             # lista di righe intere
#riga1 = riga[2:len(riga)]
lcf.close()
peak = 55932.72
mu = 39.78

redshift = 0.107
m400_obs_red = 0.039
m520_obs_red = 0.027
mjd400, mjd520, m400_abs, m520_abs, dy = [],[],[],[],[]

for line in riga:
  p = line.split()
  if float(p[1]) != 9999:
      mjd400.append((float(p[0]) - peak)/(1.+redshift))
      m400_abs.append(float(p[1]) - mu - m400_obs_red)
      dy.append(float(p[2]))
  if float(p[3]) != 9999:
      mjd520.append((float(p[0]) - peak)/(1.+redshift))
      m520_abs.append(float(p[3]) - mu - m520_obs_red)


ph400 = mjd400[3:22]
ph520 = mjd520[3:22]
dy = dy[3:22]
m400 = m400_abs[3:22]
m520 = m520_abs[3:22]
flux = 10.**(31.4 - 0.4 * np.array(m400))
dyf = np.array(dy) * 0.921034 *  flux #10.**(-0.4 * (np.array(m400)-np.array(dy)))
dyfg = dyf
#print flux, dyf, ph400
fm = max(flux)
dyf = dyf/fm
y = flux/fm#m400
X = ph400
## Mesh the input space for evaluations of the real function, the prediction and
## its MSE
x = np.atleast_2d(np.linspace(-15, 60, 1000)).T

fr400 = np.polyfit(X, y, 3)
p400 = np.poly1d(fr400)
pn400 = p400(x)
# Instanciate a Gaussian Process model
#kernel = C(10.0, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))
#kernel = C(10.0, (1e-3, 1e3))* RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
#    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
kernel = Matern(length_scale=15000, length_scale_bounds=(1e-05, 100000.0), nu=1.5)

#print 'test X before', X
X = np.atleast_2d(X).T

#print 'test y before noise', y
# Observations and noise
y = np.array(y)
#print type(X), type(y)
#dy1 = np.random.random(y.shape)
#noise = np.random.normal(0, dy)
#dy1 += noise

#print 'test y', y
#print 'test dy', dyf
#print 'test x', X

# Instanciate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, alpha= (dyf / y) ** 2, #(dyf / y) ** 2,
                              n_restarts_optimizer=500)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

print 'Parameters:', gp.get_params(deep=True)

print 'Score:', gp.score(X,y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)
#print 'Sigma:', sigma

likel = y_pred - sigma
m = max(y_pred)
mp = [i for i, j in enumerate(y_pred) if j == m]
Mag400p = -((np.log10(y_pred[mp]*fm) - 31.4)/0.4)
Mag400p_err = (-(np.log10(y_pred[mp]*fm) - 31.4)/0.4) - (-(np.log10(likel[mp]*fm) - 31.4)/0.4)

print 'Peak item, Mag, Mag-err, Peak:', mp, Mag400p, Mag400p_err, x[mp]

for a, b, c in zip(x, y_pred, likel):
    if a > (x[mp] + 9.97) and a < (x[mp] + 10.03):
        print 'Phase, Mag, Mag-err:', a-x[mp], b, c
    if a > (x[mp] + 19.96) and a < (x[mp] + 20.04):
        print 'Phase, Mag, Mag-err:', a-x[mp], b, c
    if a > (x[mp] + 29.97) and a < (x[mp] + 30.03):
        print 'Phase, Mag, Mag-err:', a-x[mp], b, c
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE

############# george part #########
#print flux, dyf, ph400
Yg = np.array(flux)
Y_err = np.array(dyfg)
Xg = np.array(ph400)

norm = Yg.max()
Yg /= norm
Y_err /= norm

gp = george.GP(Matern32Kernel(500)) # + WhiteKernel(0.001))
gp.compute(Xg, Y_err)
p0 = gp.get_parameter_vector()

def ll(p):
    gp.set_parameter_vector(p)
    return -gp.lnlikelihood(Yg, quiet=True)

def grad_ll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_lnlikelihood(Yg, quiet=True)



results = opt.minimize(ll, p0, jac=grad_ll)
#print(np.exp(gp.kernel[:]))
#print results

t = np.linspace(-15, 55, 500)
mu, cov = gp.predict(Yg, t)
std = np.sqrt(np.diag(cov))


fig = plt.figure()
plt.plot(x, pn400, 'k:', label=u'3rd order polynomial')
plt.errorbar(X.ravel(), y, dyf, fmt='.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, color='red',ls='-', label=u'S Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - sigma, #1.9600
                        (y_pred + sigma)[::-1]]),
         alpha=.8, fc='orange', ec='None', label=u'S 68$\%$ confidence interval')
plt.fill_between(t, mu+std, mu-std, color='grey', alpha=0.6,label='G 68$\%$ confidence interval')
plt.plot(t, mu,color='green',ls='-', label=u'G prediction')

plt.xlabel('Phase (day)',size=20)
plt.ylabel('Normalised flux',size=20)
#plt.ylim(-20, -22.5)
plt.legend(loc='lower left')

plt.show()

#fig.savefig('GP_test_M32K.pdf',bbox_inches='tight',format='pdf',dpi=1000)
