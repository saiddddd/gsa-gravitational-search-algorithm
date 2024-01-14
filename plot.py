import matplotlib.pyplot as plt  

def Plot(x):
    plt.semilogy(x)
    plt.title('Gravitational Searchh Algorithm : Convergence Curve')
    plt.xlabel('Iterations')
    plt.ylabel('fitness value')
    plt.show()