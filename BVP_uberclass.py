import numpy as np

class Grid:
    def __init__(self, N, a, b):
        self.N = N
        self.a = a
        self.b = b
        self.dx = (b - a) / (N - 1)
        self.x = np.linspace(a, b, N)
        
class BoundaryCondition:
    def __init__(self, kind, value):
        self.kind = kind
        self.value = value
        
    def apply(self, A, b, grid):
        if self.kind == "Dirichlet":
            A[0, :] = 0
            A[0, 0] = 1
            b[0] = self.value
            
            A[-1, :] = 0
            A[-1, -1] = 1
            b[-1] = self.value
            
        elif self.kind == "Neumann":
            A[0, 0] = -1 / grid.dx
            A[0, 1] = 1 / grid.dx
            b[0] -= self.value / grid.dx
            
            A[-1, -1] = 1 / grid.dx
            A[-1, -2] = -1 / grid.dx
            b[-1] += self.value / grid.dx
            
        else:
            raise ValueError("Unsupported boundary condition. Please choose 'Dirichlet' or 'Neumann'.")
            
class SourceTerm:
    def __init__(self, q):
        self.q = q
        
    def evaluate(self, grid):
        return self.q(grid.x[1:-1])
        
def construct_A_and_b(grid, bc_left, bc_right):
    N = grid.N
    
    # construct the matrix A
    A = np.zeros((N, N))
    for i in range(1, N-1):
        A[i, i-1] = 1 / grid.dx**2
        A[i, i] = -2 / grid.dx**2
        A[i, i+1] = 1 / grid.dx**2
        
    # apply the boundary conditions
    b = np.zeros(N)
    bc_left.apply(A, b, grid)
    bc_right.apply(A, b, grid)
    
    return A[1:-1, 1:-1], b[1:-1]