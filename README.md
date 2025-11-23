# harmonic_NN
Exploration of using NN to solve the classical simple harmonic oscillator 
This model takes in $X = (x_0, p_0, t)$ and learns to predict $Y = (x(t), p(t))$. 
In this simple form, we are trying to fit a NN to some sinusoidal functions. 
The intention here is to get my hand dirty with a simple application of NN in physics. 
This model performs well. However it clearly doesn't learn physics concepts like conservation of energy. 
Furthermore, activations of the neurons in the hidden layers is not periodic. 
The model cannot generalize for time periods outside the time interval on which it was trained.  