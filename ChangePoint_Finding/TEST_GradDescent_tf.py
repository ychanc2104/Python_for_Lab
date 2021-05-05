import numpy as np
import matplotlib.pyplot as plt
from basic.noise import normal
import tensorflow as tf

##  target function
def function(x, args):
    p1 = args[0]
    p2 = args[1]
    p3 = args[2]
    return p1*x + p2*x**2 + p3*x**3

##  aim: minimize sum square error
def loss_fun_tf(function, x, data, p_var):
    y_pred = function(x, p_var)
    y_true = data
    loss = tf.reduce_sum(tf.pow(y_pred - y_true, 2))
    return loss

##  gradient of loss function
def grad_loss_tf(loss_fun_tf, x, data, p_var):
    with tf.GradientTape() as tape:
        loss = loss_fun_tf(function, x, data, p_var)
        grad = tape.gradient(loss, p_var).numpy()
        return grad

##  initiate parameters
def init_params(grad):
    # l = 3 # learning rate
    grad = np.array(grad)
    n_params = len(grad)
    #  parameters for AdaGrad
    n = 0
    ep = 1e-8
    #  parameters for RSMprop
    alpha = 0.9
    ep = 1e-8
    sigma = np.sqrt((1-alpha) * grad**2 + ep)
    #  paraameters for Momentum
    v = np.zeros(n_params)
    lamda = 0.9
    #  paraameters for Adam
    beta1 = 0.9*np.ones(n_params)
    beta2 = 0.999*np.ones(n_params)
    m = beta1 * np.zeros(n_params) + (1 - beta1) * grad
    v_adam = beta2 * np.zeros(n_params) + (1 - beta2) * grad**2
    return n, ep, alpha, sigma, v, lamda, beta1, beta2, m, v_adam



##  simulate data with noise
x = np.arange(-9, 8, 0.05)
data = function(x, [10, 3, 1]) + normal(len(x), 0, 30)

##  preparing initial values
p = [8.0, 10.0, 0.0]
grad = grad_loss_tf(loss_fun_tf, x, data, tf.Variable(p))
converged = False
i = 0
tol = 1e-3
method = 'Adam'
n, ep, alpha, sigma, v, lamda, beta1, beta2, m, v_adam = init_params(grad)
l = 0.02 ## learning rate

p_try, grad_all = [], []
while ((not converged or i < 50) and (i < 5000)):
    with tf.GradientTape() as tape:
        if method == 'AdaGrad':
            n += grad**2
            l_t = l/np.sqrt(n + ep)
            p_new = p - l_t * grad  # update fitting parameters
        elif method == 'RSMprop':
            sigma = np.sqrt(alpha * sigma**2 + (1-alpha) * grad**2 + ep)
            p_new = p - l/sigma * grad
        elif method == 'Momentum':
            v = lamda * v - l * grad
            p_new = p + v
        else: #default is Adam
            m_hat = m / (1 - beta1)
            v_adam_hat = v_adam / (1 - beta2)
            p_new = p - l * m_hat / (np.sqrt(v_adam_hat) + ep)
            m = beta1 * m + (1 - beta1) * grad
            v_adam = beta2 * v_adam + (1 - beta2) * grad**2

        p_var = tf.Variable(p)
        grad_all += [grad_loss_tf(loss_fun_tf, x, data, p_var)]

        converged = all(abs(p_new - p) < tol)
        grad = grad_all[-1]
        i += 1
        p_try += [p]
        p = p_new


fig, ax = plt.subplots()
ax.plot(np.array(p_try))
ax.set_xlabel('iteration', fontsize=16)
ax.set_ylabel('papameters', fontsize=16)
print(f'solutions are {p_try[-1]}')

fig, ax = plt.subplots()
ax.plot(np.array(grad_all))
ax.set_xlabel('iteration', fontsize=16)
ax.set_ylabel('gradient', fontsize=16)

fig, ax = plt.subplots()
ax.plot(x, data)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('f(x)', fontsize=16)
ax.plot(x, function(x, p_try[-1]), 'r--')