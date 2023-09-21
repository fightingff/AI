# My little AI Projects for fun

***Keep Learning & gaining progress!***

- **Prisoner**(AI player for simulating *Prisoners' Dilemma*)
- **Linear Regression**(My own implementation of *Linear Regression* with *Gradient Descent*)

  - Simple implementation
  - Vectorized implementation with feature scaling(z-score)
  - Dynamic chart for visualizing the process of  gradient descent
- **Logistic Regression**

  - Sigmod function to estimate
  - Regularized
  - Scattered pictures to visualize the process
  - Ordinary Least Square
    $W =\frac{\bar{x*y} - \bar x* \bar y}{\bar {x^2} - {\bar x^2} }$
    $B = \bar Y - W * \bar{x}$
- **Network**

  - Training Network using tensorflow to do digit recognition(simple)
  - Recognize the newly created picture
- **Cluster**

  - Use K - means algorithm to do clustering (minimium $\sum ||xi-ci||$)
    1. Randomly initialize K cluster centers
    2. Repeat
       1. Cluster assignment step: assign each data point to the closest cluster center
       2. Move cluster center step: compute the mean of each cluster and assign the new mean as the cluster center
    3. Check convergence
  - Scattered pictures to notably visualize the process and the conclusion
- **Detection**

  - Construct **Normal Distribution** for each feature of the training set(all normal datas)
    $\mu_i = \frac{1}{N} \Sigma x^{(k)}_i$
    $\delta^2 = \frac{1}{N-1} \Sigma (x^{(k)}_i - \mu_i)^2$
  - Check every test vector by calulating the **product** of the probability of each feature as the final probability, and then compare it with the **threshold** to decide whether it is an anomaly or not
  - Maybe each feature is commonly not strictly independent, but the result is often still good enough
  - When features are not Gaussian enough, we can use $log(x+K)$ or $x^k(0<k<1)$ to handle it
