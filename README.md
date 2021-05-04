# Python for Gradient Descent, EM Algorithm, Fourier Transform and Image Analysis


> **Prepare environment**

* **Create virtual environment**

  1.Install

  If you have not installed virtual environment,

  `$ sudo apt-get install python3-venv`

  2.Create

  `$ python3 -m venv /path/to/virtual/environment`
  
  ex:
  `$ python3 -m venv myvenv`

  3.Enter
  
  `$ source myvenv/bin/activate`

  If succeeded,

  `(myvenv) computer@user:~$ `


* **Install required packages in venv**

  `$ pip install -r path/to/requirements.txt`

> **basic**

Put some simple and basic function, ex: binning, filer, math equation...


> **ChangePoint_Finding**

Find change point using [gradient descent algorithm][1] with 
AdaGrad, RSMprop, Momentum and Adam method.
Demonstration for derive the gradient of loss function by [tensorflow][2].


* **Gradient descent**
  
  We update our model parameters according to gradient of loss function,

  **&theta;<sub>i+1</sub> = &theta;<sub>i</sub> - &eta;&nabla;L<sub>i</sub>**
  
  where, &eta; is learning rate and &nabla;L<sub>i</sub> is 
  gradient of loss function of iteration i.
  
  An obvious drawback is that learning rate is a constant for each iteration.
  Large &eta; may cause un-stability for searching minimum.
  However, small &eta; may result in sticking in small local minimum.
  
  There are some adaptive method for dynamically adjusting learning rate.

* **AdaGrad method**

  Learning rate is inverse proportional to 
  square root of sum over past gradient of loss function.

  ![image][101]

  Accumulated sum in denominator cause learning rate eventually become 
  infinitesimally small. No longer update the parameters at the end. 
  
  ```
  n += grad**2
  l_t = l/np.sqrt(n + ep)
  p_new = p - l_t * grad  # update fitting parameters
  ```

* **RSMprop method**

  Learning rate is adapted by past gradient(weighting = &alpha;) 
  and current gradient(weighting = 1 - &alpha;).
  Therefore, learning rate is not converging to zero as fast as 
  AdaGrad method.
  ![images][102]

  Recommended &alpha; = 0.9, &eta; = 0.001.
  
  ```
  sigma = np.sqrt(alpha * sigma**2 + (1-alpha) * grad**2 + ep)
  p_new = p - l/sigma * grad
  ```


* **Momentum method**

  This is an adaptive method good for preventing algorithm sticking in
  a local minimum.
  The momentum term can be used to gain faster convergence, 
  reduce oscillation and conserve **'momentum'** 
  to skip small local minimum.

  ![images][103]

  Recommended &lambda; = 0.9  

  ```
  v = lamda * v - l * grad
  p_new = p + v
  ```

* **Adam method**

  Adaptive Moment Estimation (Adam) is generally equal to RSMprop + Momentum method.

  ![images][104]

  Recommended &beta;<sub>1</sub> = 0.9, &beta;<sub>2</sub> = 0.999

  ```
  m_hat = m / (1 - beta1)
  v_adam_hat = v_adam / (1 - beta2)
  p_new = p - l * m_hat / (np.sqrt(v_adam_hat) + ep)
  m = beta1 * m + (1 - beta1) * grad
  v_adam = beta2 * v_adam + (1 - beta2) * grad**2
  ```

* **Find gradient using tensorflow**

  We can easily derive the gradient of any loss function using tensorflow, 
  so we don't need to actually find the analytical solution of derivative loss function.
  
  ```
  ##  target function
  def function(x, args):
      p1 = args[0]
      p2 = args[1]
      p3 = args[2]
      return p1*x + p2*x**2 + p3*x**3
  
  ##  simulate data with noise
  x = np.arange(-2, 6, 0.05)
  data = function(x, [10.0, 7, 5]) + normal(len(x), 0, 100)
  
  ##  gradient
  with tf.GradientTape() as tape:
      y_pred = function(x, p_var)
      y_true = data
      ##  lose function is sum of squared error
      loss = tf.reduce_sum(tf.pow(y_pred - y_true, 2))
      grad = tape.gradient(loss, p_var).numpy()
  ```

* **Demonstrations**

  First, we model a set of data, **p<sub>1</sub>x + p<sub>2</sub>x<sup>2</sup> + p<sub>3</sub>x<sup>3</sup>**
  with p<sub>1</sub> = 10.0, p<sub>2</sub> = 3.0, p<sub>3</sub> = 1.0 and add Gaussian noise N(0, 30).
  
  ![images][105]

  Next, we use Adam method and set learning rate is 0.05. 
  Set the stopping criteria is difference of &Delta;parameters < 0.001.

  ![images][106]

  ![images][107]

  Fitting results show below,
  p<sub>1</sub> = 8.592, p<sub>2</sub> = 3.050, p<sub>3</sub> = 0.973.
  
  Because p<sub>1</sub>x term is relative small to p<sub>2</sub>x<sup>2</sup>
  and p<sub>3</sub>x<sup>3</sup>, p<sub>1</sub> fitting result slightly
  deviate from true value.
  
  ![images][108]




[1]: https://ruder.io/optimizing-gradient-descent/
[2]: https://www.tensorflow.org/guide/autodiff?hl=zh-tw


[101]: doc/img/CP/AdaGrad.png
[102]: doc/img/CP/RSMprop.png
[103]: doc/img/CP/Momentum.png
[104]: doc/img/CP/Adam.png
[105]: doc/img/CP/data.png
[106]: doc/img/CP/grad_0.05.png
[107]: doc/img/CP/params_0.05.png
[108]: doc/img/CP/data_fit.png
