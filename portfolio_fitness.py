import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

def log_return(x):
    return torch.log(x[-1])

def log_of_downside(x):
    x = x[x < 1]
    return torch.nan_to_num(-torch.sum(torch.log(x)))

def gini_t(x):
    #x = torch.Tensor(x).flatten()
    if torch.amin(x) < 0:
        x -= torch.amin(x)
    x += 0.0000001
    x, b = torch.sort(x)
    index = torch.arange(1, x.shape[0] + 1).to(device)
    n = x.shape[0]
    return ((torch.sum((2 * index - n  - 1) * x)) / (n * torch.sum(x)))

def objective_function(portfolio, weights):
    r, d, g = log_return(portfolio), log_of_downside(portfolio), gini_t(weights)
    return (r - d) * (1 - g)

def fitness(assets, weights):
    portfolio = (assets.T * weights).sum(axis = 1)
    score = objective_function(portfolio, weights)
    return score