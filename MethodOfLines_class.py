import numpy as np

class Grid:
    def __init__(self, N, a, b):
        '''Create a grid with N points between a and b.'''
        self.N = N
        self.a = a
        self.b = b
        self.dx = (b - a) / (N)
        self.x = np.linspace(a, b, N)
        
class BoundaryCondition:
    def __init__(self, kind, value, alpha=0, beta=0, gamma=0, delta=0):
        '''
        Create a boundary condition object.
        Args:
            kind: string, either "Dirichlet", "Neumann" or "Robin"
            value: float, the value of the boundary condition
            alpha, beta, gamma, delta: floats, the coefficients of the boundary condition

        Returns:
            None
        '''
        self.kind = kind
        self.value = value
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        
    def apply(self, A, b, grid):
        '''
        Apply the boundary condition to the matrix A and the vector b.
        Args:
            A: numpy array, the matrix A
            b: numpy array, the vector b
            grid: Grid object

        Returns:
            A and b are modified according to the boundary condition
        '''

        if self.kind == "Dirichlet":
            A = A[1:-1, 1:-1]
            b = b[1:-1]
            # modify A
            A[-1,-2] = 1
            # modify the last row and the last element of b
            b[0] = self.alpha # Dirichlet condition at x=a
            b[-1] = self.beta # Dirichlet condition at x=b
            return A, b
            
        elif self.kind == "Neumann":
            A = A[1:, 1:]
            b = b[1:]
            # modify A
            A[-1, -2] = 2

            # modify the first and last element of b
            b[0] = self.alpha # Dirichlet condition at x=a
            b[-1] = 2*self.delta*grid.dx # Neumann condition at x=b

            return A, b

        elif self.kind == "Robin":
            A = A[1:, 1:]
            b = b[1:]
            # modify A
            A[-1, -1] = -2*(1+self.gamma*grid.dx) 
            A[-1, -2] = 2
            # modify the first and last element of b
            b[0] -= self.alpha # Dirichlet condition at x=a
            b[-1] += self.gamma*grid.dx # Robin condition at x=b

            return A, b
            
        else:
            raise ValueError("Unsupported boundary condition. Please choose 'Dirichlet' or 'Neumann' or 'Robin'.")
        
def construct_A_and_b(grid, bc_type):
    '''
    Construct the matrix A and the vector b.

    Args:
        grid: Grid object
        bc_type: BoundaryCondition object
    
    Returns:
        A: numpy array, the matrix A
        b: numpy array, the vector b
    '''
    N = grid.N
    # construct the matrix A
    A = np.zeros((N, N))
    for i in range(0, len(A)-1):
        A[i, i-1] = 1
        A[i, i+1] = 1 
    for i in range(0, len(A)):
        A[i, i] = -2

    # apply the boundary conditions
    b = np.zeros(len(A))
    A, b = bc_type.apply(A, b, grid)
    return A, b