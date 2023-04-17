import numpy as np

class Grid:
    def __init__(self, N, a, b):
        self.N = N
        self.a = a
        self.b = b
        self.dx = (b - a) / (N - 1)
        self.x = np.linspace(a, b, N-1)
        
class BoundaryCondition:
    def __init__(self, kind, value, gamma=1):
        self.kind = kind
        self.value = value
        self.gamma = gamma
        
    def apply(self, A, b, grid):
        if self.kind == "Dirichlet":
            A[0, :] = 0
            A[0, 0] = 1
            b[0] = self.value
            
            A[-1, :] = 0
            A[-1, -1] = 1
            b[-1] = self.value
            
        elif self.kind == "Neumann":
            A[0, 0] = -1
            A[0, 1] = 1 
            b[0] -= self.value 
            
            A[-1, -1] = 1 
            A[-1, -2] = -1 
            b[-1] += self.value

        elif self.kind == "Robin": 
            b[0] -= self.value 
        
            A[-1, -1] = -2*(1+self.gamma*grid.dx) 
            A[-1, -2] = 2 
            b[-1] += self.value


            
        else:
            raise ValueError("Unsupported boundary condition. Please choose 'Dirichlet' or 'Neumann' or 'Robin'.")
        
def construct_A_and_b(grid, bc_left, bc_right, kind):
    N = grid.N
    if kind == "Dirichlet":
        # construct the matrix A
        A = np.zeros((N-1, N-1))
        for i in range(0, N-2):
            A[i, i-1] = 1
            A[i, i+1] = 1 
        for i in range(0, N-1):
            A[i, i] = -2
        
        # apply the boundary conditions
        b = np.zeros(N-1)
        bc_left.apply(A, b, grid)
        bc_right.apply(A, b, grid)
    
    elif kind == "Neumann":
        # construct the matrix A
        A = np.zeros((N, N))
        for i in range(0, N-1):
            A[i, i-1] = 1
            A[i, i+1] = 1 
        for i in range(0, N):
            A[i, i] = -2
        A[N-1, N-2] = 1
        # apply the boundary conditions
        b = np.zeros(N)
        bc_left.apply(A, b, grid)
        bc_right.apply(A, b, grid)

    elif kind == "Robin":
        # construct the matrix A
        A = np.zeros((N-1, N-1))
        for i in range(0, N-2):
            A[i, i-1] = 1
            A[i, i+1] = 1 
        for i in range(0, N-2):
            A[i, i] = -2
        
        # apply the boundary conditions
        b = np.zeros(N-1)
        bc_left.apply(A, b, grid)
        bc_right.apply(A, b, grid)
        
    else:
        raise ValueError("Unsupported boundary condition. Please choose 'Dirichlet' or 'Neumann' or 'Robin'.")
    return A[1:-1, 1:-1], b[1:-1]