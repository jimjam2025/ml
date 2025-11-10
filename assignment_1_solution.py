import math
import numpy as np
from scipy import linalg

# Assignment 1: Implementation of Python Basic Libraries

# a) Usage of methods such as floor(), ceil(), sqrt(), isqrt(), gcd() etc.
print("--- Assignment 1a ---")
print(f"Floor of 3.7 is {math.floor(3.7)}")
print(f"Floor of -2.3 is {math.floor(-2.3)}")
print(f"Ceil of 3.2 is {math.ceil(3.2)}")
print(f"Ceil of -2.7 is {math.ceil(-2.7)}")
print(f"Square root of 25 is {math.sqrt(25)}")
print(f"Square root of 2 is {math.sqrt(2)}")
print(f"Integer square root of 25 is {math.isqrt(25)}")
print(f"Integer square root of 26 is {math.isqrt(26)}")
print(f"Integer square root of 0 is {math.isqrt(0)}")
print(f"GCD of 48 and 18 is {math.gcd(48, 18)}")
print(f"GCD of 17 and 5 is {math.gcd(17, 5)}")
print("\n")

# b) Usage of attributes of array such as ndim, shape, size, methods such as sum(), mean(), sort(), sin() etc.
print("--- Assignment 1b ---")
arr = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
print(f"Array:\n{arr}")
print(f"Number of dimensions: {arr.ndim}")
print(f"Shape of array: {arr.shape}")
print(f"Size of array: {arr.size}")
print(f"Sum of array elements: {arr.sum()}")
print(f"Mean of array elements: {arr.mean()}")

# Create a new array for sorting to avoid modifying the original for other operations if needed
arr_sort = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
arr_sort.sort()
print(f"Sorted array:\n{arr_sort}")

print(f"Sine of array elements:\n{np.sin(arr)}")
print("\n")

# c) Usage of methods such as det(), eig() etc.
print("--- Assignment 1c ---")
matrix_det = np.array([[1, 2], [3, 4]])
determinant = np.linalg.det(matrix_det)
print(f"The determinant of the matrix is: {determinant}")

matrix_eig = np.array([[2, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(matrix_eig)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")
print("\n")

# d) Consider a list datatype(1D) then reshape it into2D, 3D matrix using numpy
print("--- Assignment 1d ---")
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
np_array = np.array(my_list)
print("Original 1D NumPy Array:")
print(np_array)
print("Shape:", np_array.shape)
print("\n")

matrix_2d = np_array.reshape(3, 4)
print("Reshaped 2D Matrix:")
print(matrix_2d)
print("Shape:", matrix_2d.shape)
print("\n")

matrix_3d = np_array.reshape(2, 2, 3)
print("Reshaped 3D Matrix:")
print(matrix_3d)
print("Shape:", matrix_3d.shape)
print("\n")

# e) Numpy.random.Generator and matrices using numpy
print("--- Assignment 1e ---")
rng = np.random.default_rng(seed=42)

random_floats = rng.random(size=(2, 3))
print(f"Uniform distribution (floats between 0 and 1):\n{random_floats}")

random_integers = rng.integers(low=1, high=11, size=(3, 3))
print(f"Integers within a range (1 to 10):\n{random_integers}")

normal_values = rng.normal(loc=0, scale=1, size=(2, 2))
print(f"Normal (Gaussian) distribution:\n{normal_values}")

matrix_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_b = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

product = np.dot(matrix_a, matrix_b)
print(f"Matrix multiplication (np.dot):\n{product}")

product_at = matrix_a @ matrix_b
print(f"Matrix multiplication (@ operator):\n{product_at}")

transpose_a = matrix_a.T
print(f"Transpose of matrix_a:\n{transpose_a}")

sum_matrices = matrix_a + matrix_b
print(f"Element-wise sum of matrices:\n{sum_matrices}")
print("\n")

# f) Find the determinant of a matrix using scipy
print("--- Assignment 1f ---")
matrix_A_scipy = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
determinant_A_scipy = linalg.det(matrix_A_scipy)
print("Matrix A:")
print(matrix_A_scipy)
print(f"\nDeterminant of Matrix A: {determinant_A_scipy}")
print("\n")

# g) Find eigen value and eigen vector of a matrix using scipy
print("--- Assignment 1g ---")
A_scipy = np.array([[2, 1], [1, 2]])
eigenvalues_scipy, eigenvectors_scipy = linalg.eig(A_scipy)
print("Eigenvalues:", eigenvalues_scipy)
print("Eigenvectors:\n", eigenvectors_scipy)

