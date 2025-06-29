import numpy as np

def sharpe_ratio(returns, risk_free=0):
    excess = np.array(returns) - risk_free
    if np.std(excess) == 0:
        return 0
    return np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252)

def sortino_ratio(returns, risk_free=0):
    excess = np.array(returns) - risk_free
    downside = np.std([r for r in excess if r < 0])
    if downside == 0:
        return 0
    return np.mean(excess) / (downside + 1e-8) * np.sqrt(252)

def max_drawdown(values):
    values = np.array(values)
    roll_max = np.maximum.accumulate(values)
    drawdown = (values - roll_max) / roll_max
    return drawdown.min()

def cagr(values, years):
    values = np.array(values)
    return (values[-1] / values[0]) ** (1 / years) - 1
