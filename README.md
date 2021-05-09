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

  4.Exit 
  
  `$ deactivate`


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
  data = function(x, [10.0, 7, 5]) + normal(len(x), 0, 30)
  
  ##  gradient
  with tf.GradientTape() as tape:
      y_pred = function(x, p_var)
      y_true = data
      ##  lose function is sum of squared error
      loss = tf.reduce_sum(tf.pow(y_pred - y_true, 2))
      grad = tape.gradient(loss, p_var).numpy()
  ```

* **Find solutions using gradient descent**

  We implement four adaptive learning methods below,
  
  ```
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
  ```

  Create a loop to continuously execute code above, and you 
  can get converged model parameters.

* **Demonstrations**
  
  First, we model a set of data, **p<sub>1</sub>x + p<sub>2</sub>x<sup>2</sup> + p<sub>3</sub>x<sup>3</sup>**
  with p<sub>1</sub> = 10.0, p<sub>2</sub> = 3.0, p<sub>3</sub> = 1.0 and add Gaussian noise N(0, 30).
  
  ![images][105]

  **1.Batch gradient descent**

  Next, we use Adam method and set learning rate is 0.5. 
  Set the stopping criteria is &Delta;parameters < 0.001.

  ![images][106]

  ![images][107]

  Fitting results show below,
  p<sub>1</sub> = 9.800, p<sub>2</sub> = 2.972, p<sub>3</sub> = 1.002.
  
  Because p<sub>1</sub>x term is relative small to p<sub>2</sub>x<sup>2</sup>
  and p<sub>3</sub>x<sup>3</sup>, p<sub>1</sub> fitting results slightly
  deviate from true value.
  
  ![images][108]

  Time-consuming is 11.696 seconds.

  Details see TEST_batchGradDescent.py.

  **2.mini batch gradient descent**

  [mini batch gradient descent][3] is a modified version of 
  [stochastic gradient descent(SGD)][4], which update 
  model parameters using a portion of samples instead of 
  one sample(SGD) or all samples(BGD). 
  
  Advantages are faster convergence and saving RAM consumption
  compared with typical gradient descent.

  We set batch_size = 32 and max_epoch = 150 to make iteration 
  roughly equal to iteration in 1.

  ![images][109]

  ![images][110]

  Fitting results are shown below,
  p<sub>1</sub> = 9.904, p<sub>2</sub> = 2.975, p<sub>3</sub> = 1.008.
  
  ![images][111]

  Time-consuming is 9.353 seconds.
  About 20% faster than batch gradient descent method.
  
  Details are shown in ChangePoint_Finding/TEST_mini-batchGradDescent.py.

  **3.Real case**

  We have a reaction scheme below.
  Aim is to find change-points in traces.

  **1.Simulated data with two change-points at 400 and 1000.**

  ![images][112]  

      Initial BM = 20
      
      Final BM = 70
    
      Slope = (70-20)/(1000-400) = 0.083

  **2.Norm of gradient converge to zero.**

  Secondly, use custom-built gradient descent to find change-points.
  
  The norm of gradient of each iteration are shown below,

  ![images][113]  

  Fitting results are shown below,

  ![images][114]

      Change-points = [395, 998]

      Initial BM fit = 19.8
      
      Final BM = 69.9
    
      Slope = 0.083

  Details are shown in ChangePoint_Finding/GradDescend_test_ChangePoint.py.

> **EM_Algorithm**

  [Expectation maximization][5] is an approach to implement 
  maximum likelihood estimation.

  There are two steps involved, "Expectation Step"(E-step) 
  and "Maximization Step"(M-step).
  
  **E-step: Calculate the "responsibility" of each sample i under current 
  model parameters**

  ![images][121]

  **M-step: Update model parameters using responsibilities**
  
  Example for Gaussian mixture model(GMM)

  ![images][122]

  Back to E-step until parameters converged.

* **Demonstrations**

  **1.Simulate Gaussian mixture dataset with four components**
  
  fraction = [0.25, 0.25, 0.25, 0.25], 
  
  &mu; = [6, 12, 18, 24], 
  
  &sigma; = [1, 2, 1, 2]

  ```
  import numpy as np
  import random
  
  def gen_gauss(mean, std, n_sample):
      data = []
      for m,s,n in zip(mean,std,n_sample):
          for i in range(n):
              data = np.append(data, random.gauss(m, s))
      return data
  
  ##  simulate data
  n_sample = 200
  data = gen_gauss(mean=[6,12,18,24], std=[2,1,2,1], n_sample=[n_sample]*4)
  data = data.reshape(-1,1)
  n_sample = len(data)
  ```

  **2.Automatically find components in GMM input parameter**
  
  We can apply [Bayesian information criterion(BIC)][6] or 
  [Akaike information criterion(AIC)][7] to find the most probable
  components in your dataset.
  
  Because of BIC with larger penalty term, BIC is prone to
  choose simpler model with fewer parameters.

  **3.Fit GMM**

  ```
  from EM_Algorithm.EM import *
  
  ##  fit GMM
  EMg = EM(data)
  opt_components = EMg.opt_components(tolerance=1e-2, mode='GMM', criteria='AIC', figure=False)
  
  f, m, s, converged = EMg.GMM(opt_components, tolerance=1e-2, rand_init=True)
  EMg.plot_EM_results()
  EMg.plot_fit_gauss(scatter='True')
  ```

  EM fitting converge to  
  
  fraction = [0.246, 0.250, 0.258, 0.245]
  
  mean = [5.9, 12.0, 18.1, 24.1]
  
  std = [2.2, 1.0, 2.0, 1.0]
  
  Consistent with simulated parameters.  

  ![images][123]

  Fitting results are shown below and overlay with 
  histogram.

  ![images][124]

  **4.Two-dimension data**

  We assume a situation below,

  ![images][125]

  a protein motor with two distinct movement mode,
  (X<sub>1</sub>, T<sub>1</sub>) and (X<sub>2</sub>, T<sub>2</sub>).
  
  Random variable of step size, X is Gaussian distribution and the distribution 
  of dwell time, T is f(t) = ke<sup>-kt</sup>.

  We collect(simulate) all pair of X and T.

  Model parameters below,

  n_sample = [1000, 1000]

  mean = [4, 6]

  std = [1.5, 2.0]

  tau = [2, 4]

  ```
  from EM_Algorithm.gen_gauss import gen_gauss
  from EM_Algorithm.gen_poisson import gen_poisson
  from EM_Algorithm.EM import *
  
  n_sample = 1000
  m_set = [4, 6]
  s_set = [1.5, 2]
  tau_set = [2, 4]
  data_g = gen_gauss(mean=m_set, std=s_set, n_sample=[n_sample, n_sample])
  data_p = gen_poisson(tau=tau_set, n_sample=[n_sample, n_sample])
  ```
  
  Scattering plot below,
  ![images][126]  

  Use 2-D EM to infer model parameters,

  ```
  data = np.array([data_g, data_p]).T
  EM_gp = EM(data, dim=2)
  opt_components = EM_gp.opt_components(tolerance=1e-2, mode='GPEM')
  f1, m, s1, tau, converged = EM_gp.GPEM(opt_components, tolerance=1e-2, rand_init=False)
  EM_gp.plot_EM_results()
  EM_gp.plot_gp_contour()
  ```
  Parameters converged,
  ![images][127]  
  
  Separated two joint-distribution using EM,
  ![images][128]  

  **gauss fraction is [0.58 0.42]** ([0.50, 0.50])

  **gauss center is [4.0 6.3]** ([4, 6])

  **gauss std is [1.5 1.9]** ([1.5, 2.0])
  
  **dwell time is [2.2 4.0]** ([2, 4])

    Details are shown in EM_Algorithm/EM_test_GauPoi.py.

> **Frequency analysis**

* **Fourier transform properties**

  We have a continuous signal with frequency **B** Hz.
  Then, measure with sampling rate, **f<sub>s</sub>** and acquire sample size, **N**.

  **1.Frequency resolution**
  
  Determined by number of sample size, 
  
  **&Delta;f = f<sub>s</sub>/N**.
   
  **2.Detectable bandwidth**
  
  Govern by ["Sampling Theorem"][8].
  It says **"for a given sampling rate f<sub>s</sub>, perfect reconstitution
  is guaranteed for signal with frequency, B < f<sub>s</sub>/2"**
  
  Therefore, our observed window is **0 to f<sub>s</sub>/2 Hz**.


* **Demonstrations**

  Generate signal with equally step-size = 10 in the below,

  ![images][131]
  
  According to [Wiener–Khinchin theorem][31], power spectrum can be
  derived from Fourier transform of auto-correlation function of a wide-sense-stationary random process. 
  
  **Aim: Find the step size** 

  **1.Calculate auto-correlation of spatial histogram,**

  ![images][132]

  **2.Calculate Fourier transform of auto-correlation,**
  
  Major peak is 0.1 (1/10) which consistent with simulation.
  ![images][133]
  
  Details are shown in OT/test_PSD_method.py.


> **Image analysis**

Our goals are to localize multiple beads at a time and track all objects
at each frame.

* **Objects localization**

  We generate a bead at center = (10.5, 11.6).

  ![images][141]

  **1.Find contours of objects**
  
  We use [Canny edge detector][41] to find edge of object first.

  ![images][142]

  ```
  image = np.uint8(image_tofit)
  edges = cv2.Canny(image, low, high)  # cv2.Canny(image, a, b), reject value < a and detect value > b
  contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  ```

  Parameter of high is recommended equal to 3*low.

  **2.Derive moment of contours**

  We can derive the center of mass using [image moment][42].

  ```
  def get_xy(contour):
      s = np.array(contour).shape
      m00 = s[0]
      [[m10, m01]] = np.sum(contour, axis=0)
      cX, cY = m10/m00, m01/m00
      return cX, cY
  
  for c in contours:
    x, y = get_xy(c)
  ```

  The center of position is (10.5, 11.7) which is close to (10.5, 11.6).
  
  Details are shown in TPM/TEST_contours.py.

* **Tracking all aoi with 2D-Gaussian**

  We get more accurate and precise center using two dimension Gaussian
  fitting.

  ![images][143]

  The center of position is (10.5, 11.6) which is exactly same with (10.5, 11.6).

  Details are shown in TPM/TEST_2Dgauss.py.









[1]: https://ruder.io/optimizing-gradient-descent/
[2]: https://www.tensorflow.org/guide/autodiff?hl=zh-tw
[3]: https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a
[4]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
[5]: https://ibug.doc.ic.ac.uk/media/uploads/documents/expectation_maximization-1.pdf
[6]: https://en.wikipedia.org/wiki/Bayesian_information_criterion
[7]: https://en.wikipedia.org/wiki/Akaike_information_criterion
[8]: https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
[31]: https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem

[41]: https://en.wikipedia.org/wiki/Canny_edge_detector
[42]: https://en.wikipedia.org/wiki/Image_moment

[101]: doc/img/CP/AdaGrad.png
[102]: doc/img/CP/RSMprop.png
[103]: doc/img/CP/Momentum.png
[104]: doc/img/CP/Adam.png
[105]: doc/img/CP/data.png
[106]: doc/img/CP/grad_0.5_BGD.png
[107]: doc/img/CP/params_0.5_BGD.png
[108]: doc/img/CP/data_fit_0.5_BGD.png

[109]: doc/img/CP/grad_0.5_miniBGD.png
[110]: doc/img/CP/params_0.5_miniBGD.png
[111]: doc/img/CP/data_fit_0.5_miniBGD.png
[112]: doc/img/CP/CP_trace.png
[113]: doc/img/CP/CP_norm_grad.png
[114]: doc/img/CP/CP_trace_fit.png

[121]: doc/img/EM/EM_E-step.png
[122]: doc/img/EM/EM_M-step.png
[123]: doc/img/EM/EM_progress.png
[124]: doc/img/EM/EM_results.png
[125]: doc/img/EM/Gauss_Poi_trace.png
[126]: doc/img/EM/Gauss_Poi_scatter.png
[127]: doc/img/EM/Gauss_Poi_parameters.png
[128]: doc/img/EM/Gauss_Poi_fit.png

[131]: doc/img/FT/demo_signal.png
[132]: doc/img/FT/ACF.png
[133]: doc/img/FT/PSD.png


[141]: doc/img/ImgAna/bead.png
[142]: doc/img/ImgAna/edges.png
[143]: doc/img/ImgAna/2D_Gauss.png