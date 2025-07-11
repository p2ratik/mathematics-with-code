import numpy as np
from functools import wraps

class gaussian_elimination:
    """
    A class to solve systems of linear equations using Gaussian elimination.
    
    This class represents a single linear equation in the form: ax + by + cz = d
    and provides methods to solve systems of three equations with three unknowns.
    """
    
    def __init__(self, x, y, z, const):
        """
        Initialize a linear equation with coefficients and constant term.
        
        Args:
            x (float): Coefficient of x in the equation ax + by + cz = d
            y (float): Coefficient of y in the equation ax + by + cz = d  
            z (float): Coefficient of z in the equation ax + by + cz = d
            const (float): Constant term d in the equation ax + by + cz = d
        """
        self.x = x
        self.y = y
        self.z = z
        self.const = const
    
    @staticmethod   
    def get_row(a, b, c):
        """
        Create a row vector from three coefficients.
        
        Args:
            a (float): First coefficient
            b (float): Second coefficient
            c (float): Third coefficient
            
        Returns:
            list: A list containing the three coefficients [a, b, c]
        """
        return [a, b, c]
    
    def check(func):
        """
        Decorator to check if the system is homogeneous (all constants are zero).
        
        A homogeneous system (ax + by + cz = 0) may have infinite solutions.
        
        Args:
            func: The function to be decorated
            
        Returns:
            function: Wrapped function that checks homogeneity first
        """
        @wraps(func)       
        def check_homo(*args, **kwargs):
            # Check if all constant terms are zero (homogeneous system)
            if(args[0].const==0 and args[1].const==0 and args[2].const==0):
                return ([0, 0, 0], "There may exist infinite solutions..")
            return func(*args, **kwargs)
        return check_homo
    
    @check
    def gauss(self, s1, s2):
        """
        Solve a system of three linear equations using Gaussian elimination.
        
        This method takes two additional equations and solves the system:
        self: ax + by + cz = d
        s1:   a1x + b1y + c1z = d1  
        s2:   a2x + b2y + c2z = d2
        
        Args:
            s1 (gaussian_elimination): Second equation in the system
            s2 (gaussian_elimination): Third equation in the system
            
        Returns:
            numpy.ndarray: Solution vector [x, y, z] if unique solution exists
            tuple: ([0,0,0], message) if homogeneous system with infinite solutions
        """
        # Create coefficient matrix from the three equations
        row1 = gaussian_elimination.get_row(self.x, self.y, self.z)
        row2 = gaussian_elimination.get_row(s1.x, s1.y, s1.z)
        row3 = gaussian_elimination.get_row(s2.x, s2.y, s2.z)
        
        # Convert to numpy array for matrix operations
        matrix = np.array([row1, row2, row3])
        
        # Create constant vector (right-hand side of equations)
        column = np.array([self.const, s1.const, s2.const])
        column = column.reshape((3, 1))  # Reshape to column vector
        
        try:
            # Solve the system using numpy's linear algebra solver
            soln = np.linalg.solve(matrix, column)
            return soln
        except Exception as e:
            print("Error:", e)   


# Main execution block
if __name__ == "__main__":
    # List to store the three equations
    equations = []
    
    # Input loop: Get three equations from user
    for i in range(3):
        try:
            # Get coefficients and constant for each equation
            # Format: "a b c d" where ax + by + cz = d
            val = list(map(float, input(f"Enter the values of a, b, c, d equation{i+1} in the form of : ax+by+cz = d ::").split()))  
            
            # Create equation object and add to list
            eq = gaussian_elimination(val[0], val[1], val[2], val[3])
            equations.append(eq)
        except ValueError as e:
            print("Wrong Input:", e)      
    
    # Solve the system using the first equation as reference
    # Pass the other two equations as parameters
    z = equations[0].gauss(equations[1], equations[2])
    print(z)
