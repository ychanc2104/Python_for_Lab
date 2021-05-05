from ChangePoint_Finding.TEST_batchGradDescent_tf import *

def shuffle(x, data):
    rand_index = np.arange(len(x))
    np.random.shuffle(rand_index)
    return x[rand_index], data[rand_index]

if __name__ == "__main__":
    ##  simulate data with noise
    X = np.arange(-9, 8, 0.05)
    Y = function(X, [10, 3, 1]) + normal(len(X), 0, 30)

    ##  preparing initial values
    max_epoch = 150
    batch_size = 32
    p = np.zeros(3)
    grad = grad_loss_tf(loss_fun_tf, X, Y, tf.Variable(p))
    converged = False
    method = 'Adam'
    n, ep, alpha, sigma, v, lamda, beta1, beta2, m, v_adam = init_params(grad)
    l = 0.5 ## learning rate

    X_train, Y_train = shuffle(X, Y)
    p_try, grad_all = [], []

    t_start = time.time()
    for i in range(max_epoch):
        for j in range(int(np.ceil(len(X_train)/batch_size))):
            if (i+1)*batch_size > len(X_train):
                x = X_train[j * batch_size: ]
                data = Y_train[j * batch_size: ]
            else:
                x = X_train[j*batch_size: (j+1)*batch_size]
                data = Y_train[j*batch_size: (j+1)*batch_size]

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

                grad = grad_all[-1]
                p_try += [p]
                p = p_new
    time_count = time.time() -t_start

    print(f'mini-batch Stochastic Gradient Descent spent {time_count:.3f} s')

    fig, ax = plt.subplots()
    labels = ['p$_{1}$', 'p$_{2}$', 'p$_{3}$']
    [ax.plot(np.array(p_try)[:,i], '-', label=labels[i]) for i in range(3)]
    ax.set_xlabel('iteration', fontsize=16)
    ax.set_ylabel('papameters', fontsize=16)
    ax.legend()
    print(f'solutions are {p_try[-1]}')


    fig, ax = plt.subplots()
    labels = ['p$_{1}$', 'p$_{2}$', 'p$_{3}$']
    [ax.plot(np.array(grad_all)[:,i], label=labels[i]) for i in range(3)]
    ax.set_xlabel('iteration', fontsize=16)
    ax.set_ylabel('gradient', fontsize=16)
    ax.legend()


    fig, ax = plt.subplots()
    ax.plot(X, Y)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('f(x)', fontsize=16)
    ax.plot(X, function(X, p_try[-1]), 'r--')