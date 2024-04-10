#Import required packages
import jax                                                #numpy on CPU/GPU/TPU
import sys                                                #allows for command line arguments to be run with the script
import arviz as az                                        #for saving Bayesian Data
from funcs_ub_globalfit_BL_nodop import *                          #physical recombination models (self written)
import jax.numpy as jnp                                   #jnp regularly called (ease of use)
from jax.random import PRNGKey                            #pseudo-random number generator (ease of use)
import numpyro                                            #Bayesian inference package
from numpyro.infer import MCMC, NUTS, Predictive          #MCMC with NUTS to make it Hamiltonian MC
from numpyro.distributions import TruncatedNormal, Normal #To define prior distributions
from scipy.interpolate import interp1d
import os

comp_cores = os.cpu_count()
num_chains = comp_cores-2               #number of chains to run (number of cores to use) - might need to run this on linux
numpyro.set_host_device_count(comp_cores)     #number of cores on computer
print(jax.local_device_count(),'Device Cores,' ,num_chains, 'Cores Being Used') #print number of cores being used
#Load Data - format should be a numpy file with 1st row as time, following rows as signal
data = np.load(r'MAPI_230424_Sorted/simulated_data.npy') 
#print(data.shape)
time = data[0] #time axis
ydatas = data[1:]

#print(ydatas)

#Set all floats to 64 bit
jax.config.update("jax_enable_x64", True) #Needed for precision in jax - never touch

#warm up the JIT
TRPL_HERTZ(time,jnp.array([0, 1e-15, 1e-18, 1e-3, 0.0, 1e14, 0.0]), 10.0)              #These are from the funcs_ub_globalfit file - these numbers dont mean anything, 
standardise(TRPL_HERTZ(time,jnp.array([0.0, 1e-15, 1e-18, 1e-3, 0.0, 1e14, 0.0]), 10.0)) #it is literally to get the JIT warmed up with numbers
print('JIT Warmed Up')
#Bayesian model
def model(dev, ydata = None):
    """
    
    Bayesian model for the BTDP model.

    Parameters
    ----------
    dev: float
        stdev chosen to be less than the bounds of the truncated guassians of priors (fac and theta, defined below)
    
    y0: float
        Initial counts in the TRPL signal.

    ydata: array
        standardised log10 of the experimental TRPL signal.

    """

    std_dev = dev

    N0 = 17.85

    fac = numpyro.sample(                                 # Creates a truncated normal that we use to create a distribution for the N0s later
        "fac",
        TruncatedNormal(
            low   = jnp.array([0.950, 1.950, 2.950]),
            high  = jnp.array([1.050, 2.050, 3.150]),
            loc   = jnp.array([1.000, 2.000, 3.000]), #changing the number of facs doesnt do anything
            scale = jnp.array([0.001, 0.001, 0.001]),
        ),
    ) 

    theta = numpyro.sample(                            #Truncated normal distributions of priors in special units. See NOTE for detailed conversion,
        "theta",                                       #but 1 units = 1e15 cm-3. Consider each parameters conversion with this in mind (Auger = cm6s-1)
        TruncatedNormal(
            low   = jnp.array([-38.5, -20, -20, -21, -20, 16, -0.5]), # changed the fac from 2 to 1
            high  = jnp.array([-36.5, -18, -18, -19, -18, 14, 0.5]), # and the log10 from 4 to 10
            loc   = jnp.array([-37, -19, -19, -20, -19, 15, 0]),
            scale = jnp.array([std_dev, std_dev, std_dev, std_dev, std_dev, std_dev, std_dev]),
        ),
    ) #in log form, not physically understandable - these units = 1e15 cm-3

    
    std_dev1 = 0.1
    noise = numpyro.sample(
        "noise",
        TruncatedNormal(
            low   = jnp.array([0.01, 0.01, 0.01]),
            high  = jnp.array([0.50, 0.50, 0.50]),
            loc   = jnp.array([0.10, 0.10, 0.10]), #this was a large issue - too many noise made a broadcasting error due to the final line of function
            scale = jnp.array([std_dev1, std_dev1, std_dev1]),
        ),
    )

    ka    = theta[0]
    kt    = theta[1]
    kb    = theta[2]
    kdt   = theta[3]
    kdp   = theta[4]
    NT    = theta[5]
    p0    = theta[6]

    # ka = theta[0]
    # kb = theta[1]

    #Calculate the TRPL signal and standardise
    #signal0 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]),  10**(N0 - fac[5] * jnp.log10(4.0)))
    #signal1 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]),  10**(N0 - fac[4] * jnp.log10(4.0)))
    #signal2 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]),  10**(N0 - fac[3] * jnp.log10(4.0)))
    #signal3 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]), 10**(N0 - fac[2] * jnp.log10(4.0)))
    signal4 = TRPL_HERTZ(time,jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 10**p0]), 10**N0/10**fac[1]) #changed factor to 10 from 4
    signal5 = TRPL_HERTZ(time,jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 10**p0]), 10**N0/10**fac[0]) #changed to TRPL_HERTZ from TRPL_HERTZ_HIGH added time in
    #signal4 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 0.0, 10**kb, 0.0, 0.0, 0.0, 0.0]), 10**(N0 - fac[1] * jnp.log10(4.0)))
    #signal5 = TRPL_HERTZ_HIGH(jnp.array([10**ka, 0.0, 10**kb, 0.0, 0.0, 0.0, 0.0]), 10**(N0 - fac[0] * jnp.log10(4.0)))
    signal6 = TRPL_HERTZ(time,jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 10**p0]), 10**N0)
    #signal7 = TRPL_HERTZ(time,jnp.array([10**ka, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0.0]), 10**(N0))
    
    signal = jnp.stack([signal6, signal5, signal4]) #removed signal3
    signal_s = standardise(signal)[0] #standardise the signal - this is where the incompatible broadcasting came from of [3, 1]
    #print('standardised model shape:',signal_s.shape)
    #Define the likelihood
    numpyro.sample("ydata", Normal(signal_s, noise[:, None]), obs=ydata)

num_warmup, numb_samples = 5000, 10000 #number of steps you want to do (2x samples than warmup)

rndint      = int(sys.argv[1])         #Seeds for picking of starting points in priors
rndint1     = int(sys.argv[2])         #Seeds picking from generated priors
rndint2     = int(sys.argv[3])         #Picking from posteriors
accept_prob = float(sys.argv[4])
std_dev     = float(sys.argv[5])

key1 = PRNGKey(rndint)            #Generating the random numbers from numbers above
key2 = PRNGKey(rndint1)
key3 = PRNGKey(rndint2)

#Define the MCMC - see Barney's notes for an explanation of each argument
#Adapt step size - goes around gradient of equal probability until you get back to where you started - step size and # steps will never be optimal so adjusts to get granular detail in steps
#
mcmc = MCMC(
        NUTS(model, adapt_step_size=True, max_tree_depth=6, find_heuristic_step_size=False, dense_mass=True,target_accept_prob=accept_prob),
        num_warmup=num_warmup,
        num_samples=numb_samples,
        num_chains=num_chains,
        chain_method="parallel",
        progress_bar=True,
    )

#Run the MCMC with the simulated data and noise
#ydata = np.stack((*np.log10(ydatas[:3, :60]), *np.log10(ydatas[3:, subsample_indices]))) - np.log10(ydatas.max(1))[:, None]

#ydata = np.log10(ydatas[3:, subsample_indices]) - np.log10(ydatas.max(1))[3:, None] # this is because after 6ns it was flat and alan didnt want to fit?
s_ydata, means, stds = standardise((ydatas))
#print('standardised data shape:',s_ydata.shape) 
mcmc.run(key1, dev = std_dev, ydata = s_ydata, extra_fields=["num_steps", "energy"]) #runs mcmc. Betancourt p45

mcmc.print_summary(formatter="{:.2e}") #prints stats to check how well MCMC has worked

posterior_samples = mcmc.get_samples() #gives all the numbers of posteriors resolved from MCMC
prior_predictions = Predictive(model, num_samples=numb_samples*num_chains)(key2, dev = std_dev) #makes graph of priors - changed from num_samples to numb_samples so that it doesnt mess up the num_samples between prior and posterior

posterior_predictive = Predictive(model, posterior_samples, num_samples=numb_samples*num_chains)(key3, dev = std_dev) #makes graph of posteriors


idata = az.from_numpyro(mcmc, prior = prior_predictions,
                    posterior_predictive = posterior_predictive) #save data

accept_prob = int(accept_prob * 100) #acceptance probability - accuracy of integrator
std_dev     = int(std_dev * 100)     #stdev of priors

az.to_netcdf(idata, f"fit_global_10tree_lower_ka_fixed_N0_accept{accept_prob}_std_{std_dev}_{rndint}_1")


# we feed our y data into MCMC, which has initial parameters fed in, which creates a parameter space.
#Each MCMC step calculates a signal, so we see the likelihood that the parameters at that step generate the curve we have fed in