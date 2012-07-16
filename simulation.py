from pylab import *
from pymc import *
from pymc.Matplot import plot as mplot

numquestions = 300 # number of test items being simulated
numpeople = 10 # number of participants
numthetas = 2 # number of latent proficiency variables

# toggle the following to switch between generating from/learning latent variables
# are we generating responses(1), or learning latent vars from pre-generated responses(0)?
# (note that when not generating, "correct.npy" must have been generated with the same counts as above)
generating = 0

if generating:
    # two ability params, one which alternates between -1 and 1 (across people)
    #                       and another which moves smoothly from -1 to 1
    abilities = array([linspace(-1,1,numpeople), [1,-1] * (numpeople/2)])#, list(linspace(-1,1,numpeople/2))+list(linspace(1,-1,numpeople/2))])
    # abilities = array([linspace(-1,1,numpeople), linspace(1,-1,numpeople), linspace(0,0,numpeople)])
    # abilities = array([[1]*(numpeople/3)+[0]*(numpeople/3)+[0]*(numpeople/3), [0]*(numpeople/3)+[1]*(numpeople/3)+[0]*(numpeople/3), [1]*(numpeople/3)+[0]*(numpeople/3)+[1]*(numpeople/3)])
    numpy.save("abilities", abilities)
    theta_initial = abilities
    correctness = zeros((numquestions, numpeople))
else:
    abilities = numpy.load("abilities.npy")
    theta_initial = zeros((numthetas, numpeople))
    correctness = numpy.load("correct.npy")

# theta (proficiency params) are sampled from a normal distribution
theta = Normal("theta", mu=0, tau=1, value=theta_initial, observed=generating)

# experimenting with various hyperparameters on the question discrimination parameters
# a_mu_prior = Exponential("a_mu_prior", beta=0.2, value=0)
# a_mu_prior = Normal("a_mu_prior", mu=1, tau=3, value=1)
# TODO: use inverse gamma as prior, at least for tau; makes weaker assumptions

# question-parameters (IRT params) are sampled from normal distributions (though others were tried)
# (note that the mean for the discrimination parameters isn't 0, since in general questions will be somewhat diagnostic)
a = Normal("a", mu=1, tau=1, value=[[0.0] * numthetas] * numquestions)
# a = Exponential("a", beta=0.01, value=[[0.0] * numthetas] * numquestions)
b = Normal("b", mu=0, tau=1, value=[0.0] * numquestions)

# take vectors theta/a/b, return a vector of probabilities of each person getting each question correct
@deterministic
def sigmoid(theta=theta, a=a, b=b):
    bs = repeat(reshape(b, (len(b), 1)), numpeople, 1)
    # print b.shape, a.shape, theta.shape
    return 1.0 / (1.0 + exp(bs - dot(a, theta)))

# take the probabilities coming out of the sigmoid, and flip weighted coins
correct = Bernoulli('correct', p=sigmoid, value=correctness, observed=not generating)

# create a pymc simulation object, including all the above variables
m = MCMC([a,b,theta,sigmoid,correct])

# run an interactive MCMC sampling session
m.isample(iter=20000, burn=15000)

# animated plot of theta values being sampled over time
# lines are thetas (dotted are original thetas used to generate), across the x axis is people
def plot_theta_trace():
    stepsize = 20
    trace = theta.trace()
    for i in range(0, trace.shape[0], stepsize):
        trace = theta.trace()
        sample = trace[i:i+stepsize,:,:].mean(0).T
        
        # in case we're running the simulation in another thread, wait a bit for it to catch up
        if sample[-3:,:].mean()==0:
            pause(1)
            continue
        
        clf()
        plot(abilities.T, "--")
        plot(sample)
        title(i)
        ylim((-3,3))
        pause(0.01)

if generating:
    # while correct.get_logp() > -550: a.random(); b.random(); correct.random(); print correct.get_logp()
    print correct.get_logp()
    numpy.save("correct", correct.get_value())
    print "Saved!"
else:
    plot_theta_trace()

# mplot(m)

# draw a graph of the network structure and save it to a file
# graph.graph(m).write_png("graph.png")

# matrix plot of each person's simulated % correct on each question
# imshow(mean([correct.random() for i in range(1000)],0))
