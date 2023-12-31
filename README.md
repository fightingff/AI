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
  - (3-D figures to show 2-D featured datas)
- **Recommend System**

  - **Collaborative Filtering**. Use vector $W^{(i)}$ to denote User(i)'s preferance and $X^{(j)}$ to denote Movie(j)'s features, and then use $W^{(i)} * X^{(j)} + B^{(i,j)}$ to estimate the rating of User(i) to Movie(j)
  - Like *Linear Regression*, use **Gradient Descent** to optimize the cost function
    $J(W,X,B) = \frac{1}{2} \Sigma_{marked(i,j)}(W^{(i)} * X^{(j)} +B^{(i,j)}- y^{(i,j)})^2 + \frac{\lambda}{2} \Sigma \Sigma (W^{(i,j)})^2 + \frac{\lambda}{2} \Sigma \Sigma (X^{(i,j)})^2$
  - Maybe because people's mind is so complex that we can't fit it with a simple linear function, which results in a bad performance of the model(accuracy at about only 46%)
- **Fit**

  - Build a **neural network** from scratch trying to fit a complex function given some points.
  - Every layer has a weight matrix and a bias vector, and the activation function is **sigmoid**.
  - use **gradient descent**algorithm doing **back propagation** to optimize the cost function(though the vectorized implementation is not totally mastered)
  - By some experiments, we can find that the more layers and neurals the network has, and the more epoches the model has trained, the better the complex function it can fit. Of course, it is notable that the loss can keep fluctuating even if the model has been trained for a long time.
  - Generally speaking, a small learning rate can make the model converge more steadily, but it will take more time to train the model.