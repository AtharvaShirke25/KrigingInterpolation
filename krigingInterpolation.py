import numpy as np

def readInputFile(fileName):
    """
    Read the input file and return the data as a list of lists.
    """
    data = []
    with open(fileName, 'r') as f:
        for line in f:
            data.append(line.strip().split())
            # Commented line if the line starts with '#'
            if data[-1][0] == '#':
                data.pop()
    # Convert the data to float
    for i in range(len(data)):
        data[i] = [float(data[i][0]), float(data[i][1]), float(data[i][2])]
    return data    
    
def ordinaryKriging(data, x, y):
    # Number of data points
    n = len(data)
    # Build the covariance matrix
    covarianceMatrix = np.ones((n+1, n+1))
    for i in range(n):
        for j in range(n):
            covarianceMatrix[i][j] = covariance(distance(data[i][0], data[i][1], data[j][0], data[j][1]))
            # Assign the last element/lagrange multiplier to zero
            covarianceMatrix[n][n] = 0
    # Build the right hand side vector
    rhs = np.ones(n+1)
    for i in range(n):
        dist = distance(data[i][0], data[i][1], x, y)
        rhs[i] = covariance(dist)
    # Solve the system of equations
    weights = np.linalg.solve(covarianceMatrix, rhs)
    # Calculate the interpolated value
    interpolatedValue = 0
    for i in range(n):
        interpolatedValue += weights[i] * data[i][2]
    return interpolatedValue

def covariance(h):
    """
    assume a spherical semivariogram with Nugget = 0, Sill = 1, Range a = 300
    """
    nugget, sill, range_val = 0, 1, 300
    # Formulate the spherical semivariogram
    if h < range_val:
        return nugget + sill * (1.5 * h / range_val - 0.5 * (h / range_val)**3)
    else:
        return nugget + sill


def distance(x1, y1, x2, y2):
    """
    Calculate the distance between two points.
    """
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    

def main():
    """
    Main function.
    """
    data = readInputFile('data.inp')
    output = ordinaryKriging(data, 50.0, 50.0)
    print(f'The interpolated value at (50, 50) is: {output}')

main()