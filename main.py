import csv
import numpy as np
from numpy import linalg

'''
TERESA MATTIL 100452367
RAMIRO VALDÉS 100452217

PRACTICE 1 ARTIFICIAL INTELLIGENCE
'''


def findIndex(stateWanted):
    listStates = ['High;High;High', "High;High;Low", "High;Low;High", "Low;High;High", "High;Low;Low", "Low;High;Low",
                  "Low;Low;High", "Low;Low;Low"]
    index = 0
    for k in range(0, 8):
        if listStates[k] == stateWanted:
            index = k
    return index


# Finding the desired probabilities

'''
We will find the probabilities of P( s' | s, a) which is equal to the occurrences of s',s,a divided by the occurrences of s,a
To do so, each row represents one state and each column another one. 
If we find for instance H;H;L; E ;H;L;L --> In the matrix of action E (East) at the row corresponding to state HHL and at columns of state HLL we sum 1
'''

matrixN = np.zeros((8, 8))
matrixE = np.zeros((8, 8))
matrixW = np.zeros((8, 8))

totalOccsN = [0, 0, 0, 0, 0, 0, 0, 0]
totalOccsW = [0, 0, 0, 0, 0, 0, 0, 0]
totalOccsE = [0, 0, 0, 0, 0, 0, 0, 0]

# Reading the file
with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for row in csv_reader:
        state = row[0] + ';' + row[1] + ';' + row[2]
        action = row[3]
        state2 = row[4] + ';' + row[5] + ';' + row[6]

        r = findIndex(state)
        col = findIndex(state2)

        if action == 'N':
            matrixN[r][col] += 1
            totalOccsN[r] += 1
        elif action == 'E':
            matrixE[r][col] += 1
            totalOccsE[r] += 1
        elif action == 'W':
            matrixW[r][col] += 1
            totalOccsW[r] += 1

    # Dividing it over total occurrences to Obtain Probabilities

    for j in range(0, 8):
        if totalOccsN[j] != 0:
            matrixN[j] = matrixN[j] / totalOccsN[j]
        else:
            matrixN[j] = 0
        if totalOccsE[j] != 0:
            matrixE[j] = matrixE[j] / totalOccsE[j]
        else:
            matrixE[j] = 0
        if totalOccsW[j] != 0:
            matrixW[j] = matrixW[j] / totalOccsW[j]
        else:
            matrixW[j] = 0

# Computing Bellman Equations:
V = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
states = ['High;High;High', "High;High;Low", "High;Low;High", "Low;High;High", "High;Low;Low", "Low;High;Low",
          "Low;Low;High", "Low;Low;Low"]

print(matrixW)
tol = 0.0000001  # Tolerance
diff = 1
it = 0

cN = 20  # Cost of action 'North'
cE = 20  # Cost of action 'East'
cW = 20  # Cost of action 'West'

while diff >= tol:  # Stop criteria: difference of norms between V_i and V_i+1
    x = linalg.norm(V)  # Norm of the last V vector
    it += 1
    # V(s) := min c(a) +∑_sʹ  P( sʹ| s, a) · V(sʹ) a ∈ A(s)
    for i in range(0, 8):
        # current = states[i]
        sumN = 0
        sumE = 0
        sumW = 0

        for j in range(0, 8):  # For the sum of different s', i.e, ∑ sʹP sʹ s
            # state2 = states[j]
            sumN += matrixN[i][j] * V[j]
            sumE += matrixE[i][j] * V[j]
            sumW += matrixW[i][j] * V[j]

        V[i] = min(cN + sumN, cE + sumE, cW + sumW)
        V[7] = 0

    diff = abs(linalg.norm(V) - x)


# Last iteration: Finding the optimal policies
opt = []
for i in range(0, 8):
    # current = states[i]
    sumN = 0
    sumE = 0
    sumW = 0

    for j in range(0, 7):  # Goal state Low;Low;Low doesn't have a optimal policy
        sumN += matrixN[i][j] * V[j]
        sumE += matrixE[i][j] * V[j]
        sumW += matrixW[i][j] * V[j]
    if min(cN + sumN, cE + sumE, cW + sumW) == cN + sumN:
        opt.append('N')
    elif min(cN + sumN, cE + sumE, cW + sumW) == cE + sumE:
        opt.append('E')
    elif min(cN + sumN, cE + sumE, cE + sumW) == cW + sumW:
        opt.append('W')
    else:
        opt.append('e')  # Error

print("\n******* Expected Values ******\n")
for q in range(0, 8):
    print("Expected value of state", states[q], "\tis \t ",  V[q])

print("\n******* Optimal policies ******\n")
for q in range(0, 7):
    print("Optimal policy of state", states[q], "\tis \t ",  opt[q])
print("Optimal policy of state", states[7], "\tis \t  Not defined")
