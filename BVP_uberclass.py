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
    def __init__(self, kind, value, alpha=0, beta=0, gamma=0, delta=0):
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
            A = np.delete(A, 0, 1)

            # delete the first element of b
            b = np.delete(b, 0)
            # modify A
            A[-1,-2] = 1
            # modify the last row and the last element of b
            b[0] = self.alpha
            b[-1] = self.beta
            return A, b
            
        elif self.kind == "Neumann":
            # modify A
            A[-1, -2] = 2

            # modify the first and last element of b
            b[0] = self.alpha
            b[-1] = 2*self.delta*grid.dx

            return A, b

        elif self.kind == "Robin":
            # delete the first row and the last column of A
            A = np.delete(A, 0, 0)
            A = np.delete(A, 0, 1)

            # delete the first element of b
            b = np.delete(b, 0)
            # modify A
            A[-1, -1] = -2*(1+self.gamma*grid.dx) 
            A[-1, -2] = 2
            # modify the first and last element of b
            b[0] -= self.alpha
            b[-1] += self.gamma*grid.dx

            return A, b
            
        else:
            raise ValueError("Unsupported boundary condition. Please choose 'Dirichlet' or 'Neumann' or 'Robin'.")
        
def construct_A_and_b(grid, bc_left, bc_right):
    N = grid.N
    # construct the matrix A
    A = np.zeros((N-1, N-1))
    for i in range(0, len(A)-1):
        A[i, i-1] = 1
        A[i, i+1] = 1 
    for i in range(0, len(A)):
        A[i, i] = -2

    # apply the boundary conditions
    b = np.zeros(len(A))
    A, b = bc_left.apply(A, b, grid)
    A, b = bc_right.apply(A, b, grid)
    return A,b