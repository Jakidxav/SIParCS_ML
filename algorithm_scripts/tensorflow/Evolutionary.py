'''
file for using evolutionary algorithms to train neural networks instead of backprop


Evolutionary algorithm in a nutshell:
    Create a random population of X individuals. (100 - 500)
        Each individual has a set of ‘genes.’
        each gene is a parameter in the network. Ie learning rate, momentum, network size, LSTM look back, etc
        Each gene should be randomized within reasonable limits (if applicable)
    For each individual:
        Create a corresponding network whose parameters are the genes.
        Predict on training set (need dev set? Will look into. Not having a dev set might be nice because then we can train using CV/ will have a larger training set….)

        Breed best 50% (decide on what optimization metric)
            Need to decide best breeding method. Will probably just randomly select genes from each parent giving a slightly higher probability to parents with better predictions.
        Slightly mutate genes in child/ren
        Predict on children. Add children to population.
        Keep only best individuals (doesn’t matter if original or offspring) and throw out the rest.
    Repeat from step 2 for some number of iterations

'''
