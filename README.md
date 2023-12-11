# RL 2023 Optimal Execution

This repository contains a code implementation of the Research project **Optimal Trade Execution using RL approach**.

## Idea:
This project aims to tackle the challenging problem of optimizing the execution of large orders in financial markets, whether they are simulated environments or real-world platforms like Binance. Efficiently executing large orders while minimizing costs and market impact is crucial for institutional investors and traders. Reinforcement Learning provides a promising approach to address this problem by learning optimal execution strategies in complex market dynamics.

## Simulator:

$$dQ_v^t = -v_tdt$$

$$dS_v^t = -bv_tdt+\sigma dW_t$$

$$c_v^t =(((S_v^t - k\sqrt {v_t}) - t|v_t|)v_t -\rho(Q_t^v)^2)dt$$

where $Q_v$ is asset volume, $S_v$ is return by GMB process,  $v_t$ - action, $c_v^t$ - cash, $k$ - temporary price impact coef, $b$  -permanent price impact coef, $\rho$ - inventory risk coef

In real market we assume $b$ = 0 and instead of $k\sqrt {v_t}$ compute real price decrease by the order book. What is more, high 5 bid and asks are taken as observations and reasonably real market price $S_v^t$ is taken from market, as midprice from best ask and bid. 

## Running cost

$$r_{t+1} = -dc_v^{t+1} + (Q_v^t<0)$$ 

## Repo description

* [`actor_critic.ipynb`](actor_critic.ipynb) or [`actor_critic.py`](actor_critic.py) — running the actor-critic model on simulated data 

* [`actor_critic_historical.ipynb`](actor_critic_historical.ipynb) or [`actor_critic_historical.py`](actor_critic_historical.py)— running the actor-critic model on real Binance data 

* [`actor_critic_params.yaml`](actor_critic_params.yaml) and [`actor_critic_historical_params.yaml`](actor_critic_historical_params.yaml)— hyperparams for models and simulations

* [`critic.py`](lib/critic.py) — critic arch, including TD objective

* [`model.py`](lib/model.py) — model perceptron, including its layers, architecture and etc

* [`policy.py`](lib/policy.py) — actor arch, including surrogate objective

* [`simulator.py`](lib/simulator.py) — Euler–Maruyama integrator

* [`system.py`](lib/system.py) — systems for simulating data or processing real historical data

* [`util.py`](lib/util.py) — usefull functions: data buffers, optimizers and etc


## Prerequisites
```bash
git clone https://github.com/ooodnakov/RL-2023-TP-Optimal-Execution
cd RL-2023-TP-Optimal-Execution
pip install -r requirements.txt
wget https://dnakov.ooo/files/data.parquet
```

## Running
To reproduce results, you just ran our notebooks using Python kernel with neccesary packages. 

The code was tested in `Python 3.9.16`. The code execution in other Python versions is not guaranteed.

If you prefere `.py` files you can run them via:

```bash
python actor_critic.py actor_critic_params.yaml
```
```bash
python actor_critic_historical.py actor_critic_historical_params.yaml data.parquet
```

## Results

We succeded to beat TWAP results in execution in both simulated and real markets.

### Simulated market

![alt text](/pics/cost_by_iteration.png)

![alt text](/pics/states.png)

![alt text](/pics/actions.png)

RL allowed to receive more than `33%` more cash compared to TWAP baseline

### Real market

![alt text](/pics/cost_by_iteration_real.png)

![alt text](/pics/states_real.png)

![alt text](/pics/actions_real.png)