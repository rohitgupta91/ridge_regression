Fast gradient algorithm to solve L2 regularized logistic regression problem.

The l2 regularized logistic problem writes as:

![first equation](http://latex.codecogs.com/gif.latex?min_%7B%7D%20%28%5Cbeta%29%20%3A%3D%201/n%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dlog%281%20&plus;%20exp%28-y_ix_i%5ET%5Cbeta%20%29%29%20&plus;%20%5Clambda%20%5Cleft%20%5C%7C%20%5Cbeta%20%5Cright%20%5C%7C_2%5E2)

The algorithm performs backtracking line search to determine the ideal step size to use.
___
The fast-gradient algorithm writes as:
-
![fastgrad1](http://latex.codecogs.com/gif.latex?%5Cboldsymbol%7B%5Cmathbf%7Binput%7D%7D%20%5Ctextup%7B%20step-size%20%7D%20%5Ceta%20_0%2C%20%5Ctextup%7Btarget%20accuracy%20%7D%20%5Cvarepsilon)

![fastgrad2](http://latex.codecogs.com/gif.latex?%5Ctextbf%7Binitialization%20%7D%20%5Cbeta%20_0%20%3D%200%2C%20%5CTheta%20_0%20%3D%200)

![fastgrad3](http://latex.codecogs.com/gif.latex?%5Ctextbf%7Brepeat%20%7D%20%5Ctextrm%7Bfor%20t%7D%20%3D%200%2C1%2C2%2C...)

![fastgrad4](http://latex.codecogs.com/gif.latex?%5Ctextrm%7BFind%20%7D%20%5Ceta%20_t%20%5Ctextrm%7B%20with%20backtracking%20rule%7D)

![fastgrad5](http://latex.codecogs.com/gif.latex?%5Cbeta%20_t_&plus;_1%20%3D%20%5CTheta%20_t%20-%20%5Ceta%20_t%5CUpsilon%20F%28%5CTheta%20_t%29%5Ctextrm%7B%20where%20%7D%20%5CUpsilon%20%5Ctextrm%20%7B%20is%20the%20gradient%7D)

![fastgrad6](http://latex.codecogs.com/gif.latex?%5CTheta%20_t_&plus;_1%20%3D%20%5Cbeta%20_t_&plus;_1%20&plus;%20t/%28t&plus;3%29%28%5Cbeta%20_t_&plus;_1%20-%20%5Cbeta_t%29)

![fatgrad7](http://latex.codecogs.com/gif.latex?%5Ctextbf%7Buntil%20%7D%20%5Ctextrm%7Bthe%20stopping%20criterion%20%7D%5Cleft%20%5C%7C%20%5CUpsilon%20F%20%5Cright%20%5C%7C%5Cleq%20%5Cvarepsilon)
___
Initializing Parameters:
-
- Regularization Term
- Initial step size
- Maximum Iterations
- Tolerance value

___
Python libraries needed for running the source and demo:
- numpy
- pandas
- sklearn
- random

The class labels should be given values of [-1, +1]
Examples to use the code is provided in the demo folder.
