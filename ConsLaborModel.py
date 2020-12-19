import numpy as np

def in_ipynb():
    try:
        if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
            return True
        else:
            return False
    except NameError:
        return False

if in_ipynb():
    # %matplotlib inline generates a syntax error when run from the shell
    # so do this instead
    get_ipython().run_line_magic('matplotlib', 'inline') 
else:
    get_ipython().run_line_magic('matplotlib', 'auto') 
    
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex= True)
plt.rc('text', usetex= True)
plt.rc('font', family='serif')

import warnings
warnings.filterwarnings("ignore")

import sys
import os

# Find pathname to this file:
my_file_path = os.path.dirname(os.path.abspath("LiqConstr.ipynb"))
figures_dir = os.path.join(my_file_path, "Figures") # Relative directory for primitive parameter files

sys.path.insert(0, figures_dir)
sys.path.insert(0, my_file_path)

from HARK.ConsumptionSaving.ConsLaborModel import (
    LaborIntMargConsumerType,
    init_labor_lifecycle,
)
import numpy as np
import matplotlib.pyplot as plt
from time import process_time

mystr = lambda number: "{:.4f}".format(number)  # Format numbers as strings

do_simulation = True

# Solve a labor intensive margin consumer i.e. a consumer with utility for leisure
LaborIntMargExample = LaborIntMargConsumerType(verbose=0)
LaborIntMargExample.cycles = 0

t_start = process_time()
LaborIntMargExample.solve()
t_end = process_time()
print(
    "Solving a labor intensive margin consumer took "
    + str(t_end - t_start)
    + " seconds."
)

t = 0
bMin_orig = 0.0
bMax = 100.0

#1 Plot the consumption function at various transitory productivity shocks
TranShkSet = LaborIntMargExample.TranShkGrid[t]
bMin = bMin_orig
B = np.linspace(bMin, bMax, 300)
bMin = bMin_orig
f = plt.figure()    
for Shk in TranShkSet:
    B_temp = B + LaborIntMargExample.solution[t].bNrmMin(Shk)
    C = LaborIntMargExample.solution[t].cFunc(B_temp, Shk * np.ones_like(B_temp))
    plt.plot(B_temp, C)
    bMin = np.minimum(bMin, B_temp[0])
plt.xlabel("Beginning of period bank balances")
plt.ylabel("Normalized consumption level")
plt.xlim(bMin, bMax - bMin_orig + bMin)
plt.ylim(0.0, None)
plt.show()
f.savefig(os.path.join(figures_dir, 'conslabor1.pdf'))
f.savefig(os.path.join(figures_dir, 'conslabor1.png'))
f.savefig(os.path.join(figures_dir, 'conslabor1.svg'))

#2 Plot the marginal consumption function at various transitory productivity shocks
TranShkSet = LaborIntMargExample.TranShkGrid[t]
bMin = bMin_orig
B = np.linspace(bMin, bMax, 300)
f = plt.figure() 
for Shk in TranShkSet:
    B_temp = B + LaborIntMargExample.solution[t].bNrmMin(Shk)
    C = LaborIntMargExample.solution[t].cFunc.derivativeX(
        B_temp, Shk * np.ones_like(B_temp)
    )
    plt.plot(B_temp, C)
    bMin = np.minimum(bMin, B_temp[0])  
plt.xlabel("Beginning of period bank balances")
plt.ylabel("Marginal propensity to consume")
plt.xlim(bMin, bMax - bMin_orig + bMin)
plt.ylim(0.0, 1.0)
plt.show()
f.savefig(os.path.join(figures_dir, 'conslabor2.pdf'))
f.savefig(os.path.join(figures_dir, 'conslabor2.png'))
f.savefig(os.path.join(figures_dir, 'conslabor2.svg'))


#3 Plot the labor function at various transitory productivity shocks
TranShkSet = LaborIntMargExample.TranShkGrid[t]
bMin = bMin_orig
B = np.linspace(0.0, bMax, 300)
f = plt.figure() 
for Shk in TranShkSet:
    B_temp = B + LaborIntMargExample.solution[t].bNrmMin(Shk)
    Lbr = LaborIntMargExample.solution[t].LbrFunc(B_temp, Shk * np.ones_like(B_temp))
    bMin = np.minimum(bMin, B_temp[0])
    plt.plot(B_temp, Lbr)  
plt.xlabel("Beginning of period bank balances")
plt.ylabel("Labor supply")
plt.xlim(bMin, bMax - bMin_orig + bMin)
plt.ylim(0.0, 1.0)
f.savefig(os.path.join(figures_dir, 'conslabor3.pdf'))
f.savefig(os.path.join(figures_dir, 'conslabor3.png'))
f.savefig(os.path.join(figures_dir, 'conslabor3.svg'))

#4 Plot the marginal value function at various transitory productivity shocks
pseudo_inverse = True
TranShkSet = LaborIntMargExample.TranShkGrid[t]
bMin = bMin_orig
B = np.linspace(0.0, bMax, 300)
f = plt.figure()   
for Shk in TranShkSet:
    B_temp = B + LaborIntMargExample.solution[t].bNrmMin(Shk)
    if pseudo_inverse:
        vP = LaborIntMargExample.solution[t].vPfunc.cFunc(
            B_temp, Shk * np.ones_like(B_temp)
        )
    else:
        vP = LaborIntMargExample.solution[t].vPfunc(B_temp, Shk * np.ones_like(B_temp))
    bMin = np.minimum(bMin, B_temp[0])
    plt.plot(B_temp, vP)
plt.xlabel("Beginning of period bank balances")
if pseudo_inverse:
    plt.ylabel("Pseudo inverse marginal value")
else:
    plt.ylabel("Marginal value")
plt.xlim(bMin, bMax - bMin_orig + bMin)
plt.ylim(0.0, None)
plt.show()
f.savefig(os.path.join(figures_dir, 'conslabor4.pdf'))
f.savefig(os.path.join(figures_dir, 'conslabor4.png'))
f.savefig(os.path.join(figures_dir, 'conslabor4.svg'))



