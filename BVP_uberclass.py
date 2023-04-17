import numpy as np

class Grid:
    def __init__(self, N, a, b):
        '''Create a grid with N points between a and b.'''
        self.N = N
        self.a = a
        self.b = b
        self.dx = (b - a) / (N-1)
        self.x = np.linspace(a, b, N-1)
        
class BoundaryCondition:
    def __init__(self, kind, value, alpha=1, beta=1, gamma=1, delta=1):
        '''Parameters:
        kind: string, either "Dirichlet", "Neumann" or "Robin"
        value: float, the value of the boundary condition
        alpha, beta, gamma, delta: floats, the coefficients of the boundary condition
        '''
        self.kind = kind
        self.value = value
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        
    def apply(self, A, b, grid):
        '''Parameters:
        A: numpy array, the matrix A
        b: numpy array, the vector b
        grid: Grid object

        Output:
        A and b are modified according to the boundary condition
        '''

        if self.kind == "Dirichlet":
            # delete the first row and the last column of A
            A = np.delete(A, 0, 0)
            A = np.delete(A, -1, 1)
            print(A.shape)

            # delete the first element of b
            b = np.delete(b, 0)

            # modify the last row and the last element of b
            b[0] = self.alpha
            b[-1] = self.beta
            
        elif self.kind == "Neumann":
            A[-1, -2] = 2

            b[0] = self.alpha
            b[-1] = 2*self.delta*grid.dx

        elif self.kind == "Robin":
            
            A[-1, -1] = -2*(1+self.gamma*grid.dx) 
            A[-1, -2] = 2
            b[0] -= self.alpha
            b[-1] += self.gamma*grid.dx
            
        else:
            raise ValueError("Unsupported boundary condition. Please choose 'Dirichlet' or 'Neumann' or 'Robin'.")
        
def construct_A_and_b(grid, bc_left, bc_right):
    N = grid.N
    # construct the matrix A
    A = np.zeros((N-1, N-1))
    for i in range(0, N-2):
        A[i, i-1] = 1
        A[i, i+1] = 1 
    for i in range(0, N-1):
        A[i, i] = -2

    # apply the boundary conditions
    b = np.zeros(N-1)
    A = bc_left.apply(A, b, grid)
    print(A)
    bc_right.apply(A, b, grid)
    print(A)
    return A[0:-1, 0:-1], b[0:-1]