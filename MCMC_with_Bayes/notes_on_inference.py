import plotly as plot
import plotly.graph_objs as graph
from math import factorial, e, sqrt, pi
import numpy as np
from scipy.special import gamma

#https://www.datascience.com/blog/introduction-to-bayesian-inference-learn-data-science-tutorials by Aaron Kramer
#https://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm
#https://www.amazon.com/Probability-Theory-Science-T-Jaynes/dp/0521592712

"""
0. My situation - what we need bayes for?
1. Creating the data.
2. Summarising and plotting the data.
3. Using Markov Chain Monte Carlo to get what I want.
"""

"""ad. 0
There are numerous situations when I have some
belief, or expectation on how things should be,
but then I see some evidence which changes my
belief - changes my prior belief. However the open
question is: what is the amount of belief i need
to change?

This case: I have some knowledge on historical
effectiveness of my facebook ads. I run another ad 
and I see different data - how should my expectations
about ads effectiveness change given new data?
"""

"""1a. I'll need some definitions first:"""

data_output_dir = "C:/marcin/statystyka/wykresy/" #directory where i will save my plots


def rysuj_wykres_kropki(dane, dziedzina, nazwa):
    """Given the data and chart file name,
    method automates plotting simple scatter chart"""
    trace_A = graph.Scatter(
        x = dziedzina,
        y = dane,
        mode = 'markers'
    )
    plot_data = [trace_A]

    figure = graph.Figure(
        data=plot_data
    )

    plot.offline.plot(figure, filename=data_output_dir + nazwa + '.html', auto_open=False)


def rysuj_dwa_wykresy_kropki(dane_A, dane_B, dziedzina, nazwa):
    """Given the data about to two series and chart file name,
    method automates plotting two parallel charts"""
    trace_A = graph.Scatter(
        x = dziedzina,
        y = dane_A,
        mode = 'markers'
    )
    trace_B = graph.Scatter(
        x=dziedzina,
        y=dane_B,
        mode='markers'
    )
    plot_data = [trace_A, trace_B]

    figure = graph.Figure(
        data=plot_data
    )

    plot.offline.plot(figure, filename=data_output_dir + nazwa + '.html', auto_open=False)


def rysuj_trzy_wykresy_kropki(dane_A, dane_B, dane_C, dziedzina, nazwa):
    """Given the data about to three series and chart file name,
    method automates plotting three parallel charts"""
    trace_A = graph.Scatter(
        x = dziedzina,
        y = dane_A,
        text = "posterior belief",
        mode = 'markers'
    )
    trace_B = graph.Scatter(
        x=dziedzina,
        y=dane_B,
        text="evidence distribution",
        mode='markers'
    )
    trace_C = graph.Scatter(
        x=dziedzina,
        y=dane_C,
        text="prior belief",
        mode='markers'
    )
    plot_data = [trace_A, trace_B, trace_C]

    figure = graph.Figure(
        data=plot_data
    )

    plot.offline.plot(figure, filename=data_output_dir + nazwa + '.html', auto_open=False)


def binomial_likelihood(P, trials, successes):
    """Given: P - probability of success,
    trials - number of trials,
    successes - number of successes
    returns calculated binomial likelihood"""
    return (factorial(trials) / (factorial(successes) * factorial(trials - successes))) * (P ** successes) * ((1 - P) ** (trials - successes))


def beta_likelihood(x, alpha, beta):
    """Given a data point x, and two parameters
    alpha and beta returns a corresponding value
    of beta distribution"""
    B = (gamma(alpha) * gamma(beta)) / gamma(alpha + beta)
    return ((x**(alpha - 1))*((1 - x)**(beta - 1)))/B


def beta_distribution(alpha, beta, data):
    """For a list of data points
    returns a list of new points which
    are beta distributed"""
    output = [beta_likelihood(x, alpha, beta) for x in data]
    return output


def normal_likelihood(x, mi, sigma):
    """Taking a point of a dataset,
    average of that dataset and
    its variance returns value
    of gaussian function for that point"""
    return (1 / sqrt(2*pi*sigma**2))**(e**(0 - ((x - mi)**2) / 2*sigma**2))


def get_average(data):
    """Returns the average value
    from given data points."""
    return sum(data) / len(data)


def get_std(data):
    """For a list of data
    returns its standard deviation"""
    data_average = get_average(data)
    return sqrt(sum([(x - data_average)**2 for x in data]) / (len(data) - 1))


def markovChain_metropolis(N, evidence_succes, evidence_trials, pastData_alpha, pastData_beta):
    """Returns a list of length [N]
    made by proposed points
    which lie in probability space
    described by two probability density functions (pdf).
    In this particular case those functions are
    pmf of binomial distribution and pdf
    of normal distribution
    N - number of points to be proposed,
    alpha, beta - parameters of binomial distribution"""
    current = np.random.uniform(0, 100) / 100 #let me randomly pick the first proposed click through rate in <0, 1>
    samples = [] #in this list i'll save accepted CTRs
    samples.append(current) #first CTR is proposed despite how bad could it be
    accept_ratios = []  #debugging
    for i in range(N):
        candidate = np.random.uniform(0, 100) / 100 #let me randomly pick next proposed click through rate in <0, 1>
        likelihood_current =  binomial_likelihood(current, evidence_trials, evidence_succes)
        likelihood_proposal = binomial_likelihood(candidate, evidence_trials, evidence_succes)
        prior_current =  beta_likelihood(current, pastData_alpha, pastData_beta)
        prior_proposal = beta_likelihood(candidate, pastData_alpha, pastData_beta)
        #Above I'm checking how particular candidate performs in terms of considered distributions


        probability_current = likelihood_current * prior_current
        probability_proposal = likelihood_proposal * prior_proposal
        accept = probability_proposal / probability_current
        #Above

        if accept > 1 or accept > np.random.uniform(0, 1):
            current = candidate
            samples.append(current)
            accept_ratios.append(accept)

    return samples


"""1b. Creating historical data:"""
faked_param_a = 11.5
faked_param_b = 48.5
N = 100
#I'll generate faked data for performance of several
#ad campaigns (a 100 of them).
#Let it be beta distributed with most chances
#to have click through rates somewhere around 0.2
#The beta distribution is a 2 parameter (α, β) distribution that
#is often used as a prior for the θ parameter of the binomial distribution
fake_history_probabilities = np.random.beta(faked_param_a, faked_param_b, size=N)
fake_nr_impressions = np.random.randint(1, 1000, size=N) #let me randomize number of times particular ad was seen
fake_nr_clicks = np.random.binomial(fake_nr_impressions, fake_history_probabilities) #let me randomize number of clicks for each ad

"""1c. Creating the new ad data - the new evidence:"""
nr_clicks = np.random.randint(10, 15)  #let me randomize number of clicks for new ad
nr_impressions = np.random.randint(20, 30) #let me randomize number of times new ad was seen
click_through_rates_evidence = float(nr_clicks) / nr_impressions #CTR

"""2. Summarising and plotting the data:"""
from_zero_to_one = [i/100 for i in range(100)] #those are my proposed probabilities, to be checked
past_click_through_rates = fake_nr_clicks / fake_nr_impressions

historic_count, bins = np.histogram(past_click_through_rates, bins=from_zero_to_one)
normalised_historic_histogram = [float(x) / 100 for x in historic_count] #normalising the histogram
#I need to "bin" the range of values - that is,
#divide the entire range of values into a series of intervals
#and then count how many values fall into each interval.

distribution_ctr_evidence = [binomial_likelihood(p, nr_impressions, nr_clicks) for p in from_zero_to_one]
#Above, I use binomial distribution to know for what probabilities
#in ragne from 0.01 to 1 there are how much chances that this
#particular probability is my true probability, given particular number
#of clicks and particular number of impressions.

rysuj_dwa_wykresy_kropki(normalised_historic_histogram, distribution_ctr_evidence, from_zero_to_one, "porownanie_rozkladow")
#For now, I can see the distribution of probabilty for my new evidence
#and normalised histogram points telling me what was the chances
#to see particular performance for previous ads.

#In point no. 1 I created (faked) my past experience about ads performance.
#In fact, I know how are they distributed - in real world, however
#i wouldn't know. Since that now i will find the distribution of
#this simulated past data. Having both distributions, i will be able
#to continue with the process of changing my belief
#about ad efficiency.

historicData_mean = sum(past_click_through_rates) / len(past_click_through_rates)
historicData_variance = sqrt(sum([(x - historicData_mean)**2 for x in past_click_through_rates]) / (len(past_click_through_rates) - 1))
historicData_parameter_alpha = historicData_mean**2 * ((1-historicData_mean) / historicData_variance - (1 / historicData_mean))
historicData_parameter_beta = historicData_parameter_alpha * (1 / historicData_mean - 1)
#https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters

#If I used SciPy library for this:
#pastData_parameters = beta.fit(fake_nr_clicks, floc=0, fscale=1)
#parameter_alpha, parameter_beta = pastData_parameters[0:2]


"""3. Updating my posterior belief with MCMC:"""

#Knowing how distributed are previous data and new one,
#I want to know how much should I change my belief about
#my ad campaigns - how much the new evidence really matter.
#The tool just right for this task is Bayes Rule.
#  "In probability theory and statistics, Bayes’ theorem
#  (alternatively Bayes’ law or Bayes' rule, also written as Bayes’s theorem)
#  describes the probability of an event, based on prior knowledge of conditions
#  that might be related to the event." - Wikipedia
#That's exactly what i want to know - the probability
#of an event of clicking the ad, based on prior knowledge of
#historic ad's data given some new evidence.

propositions = markovChain_metropolis(1000, nr_clicks, nr_impressions, historicData_parameter_alpha, historicData_parameter_beta)

markovSamples_count, bins = np.histogram(propositions, bins=from_zero_to_one)

normalised_markovSamples_count = [float(x) / 100 for x in markovSamples_count] #normalising the histogram

rysuj_trzy_wykresy_kropki(normalised_markovSamples_count, distribution_ctr_evidence, normalised_historic_histogram, from_zero_to_one, "final")