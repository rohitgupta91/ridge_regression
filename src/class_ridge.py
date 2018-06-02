#importing modules
import numpy as np

# defining class for ridge regression with fast gradient algorithm
class ridge_regression:
    def __init__ (self, lambda_val = 0.01, t_init = 1, max_iter = 100, eps = 0.001):
        self.lambda_val = lambda_val
        self.t_init = t_init
        self.max_iter = max_iter
        self.eps = eps
        
    #writing the objective function
    def objective_func(self, x_mat, y_array, beta_array, lambda_val):
        yhat = np.matmul(self.x_mat, beta_array)
        logterm = np.log(1 + np.exp(-(self.y_array)*(yhat)))
        res = np.mean(logterm) + self.lambda_val*np.linalg.norm(beta_array)**2
        return res
    
    #function for computing the p matrix
    def compute_p(self, x_mat, y_array, beta_array):
        rand = np.exp(-self.y_array*(self.x_mat.dot(beta_array)))
        p_val = rand/(1+rand)
        p_val = np.diag(p_val)
        return (p_val)

    #writing the function computegrad
    def computegrad(self, x_mat, y_array, beta_array, lambda_val):
        p_mat = self.compute_p(self.x_mat, self.y_array, beta_array)
        res = 2*self.lambda_val*beta_array - ((self.x_mat.T.dot(p_mat)).dot(self.y_array))/len(self.y_array)
        return res

    #writing the function backtracking
    def bt_line_search(self, x_mat, y_array, beta_array, lambda_val, t=1, alpha = 0.5, beta = 0.5, max_iter = 100):
        """
        Perform backtracking line search
        Inputs:
          - x: Current point
          - t: Starting (maximum) step size
          - alpha: Constant used to define sufficient decrease condition
          - beta: Fraction by which we decrease t if the previous t doesn't work
          - max_iter: Maximum number of iterations to run the algorithm
        Output:
          - t: Step size to use
        """
        grad_beta = self.computegrad(self.x_mat, self.y_array, beta_array, self.lambda_val)
        norm_grad_beta = np.linalg.norm(grad_beta)

        found_t = False
        i = 0  # Iteration counter
        while (found_t is False and i < self.max_iter):
            # INSERT THE SUFFICIENT DECREASE CONDITION FOR BACKTRACKING LINE SEARCH IN THE
            # if STATEMENT BELOW.      
            if (self.objective_func(self.x_mat, self.y_array, (beta_array - t*grad_beta), self.lambda_val) < 
                self.objective_func(self.x_mat, self.y_array, beta_array, self.lambda_val) - alpha*t*norm_grad_beta**2):
                found_t = True
            else:
                # INSERT THE UPDATE TO t HERE
                t *= beta
                i += 1
                #print (t)

        return t

    # writing the fastgradalgo method
    def fit(self, x_mat, y_array):
        self.x_mat = x_mat
        self.y_array = y_array
                
        self.beta_val = np.zeros(shape=(self.x_mat.shape[1]))
        theta_val = np.zeros(shape=(self.x_mat.shape[1]))
        grad_beta = self.computegrad(self.x_mat, self.y_array, self.beta_val, self.lambda_val)
        iter_v = 0
        
        while np.linalg.norm(grad_beta) > self.eps and iter_v < self.max_iter:
            t = self.bt_line_search(self.x_mat, self.y_array, self.beta_val, self.lambda_val, t = self.t_init)
            beta_old = self.beta_val
            self.beta_val = theta_val - t*self.computegrad(self.x_mat, self.y_array, theta_val, self.lambda_val)
            theta_val = self.beta_val + (iter_v/(iter_v + 3))*(self.beta_val - beta_old)
            grad_beta = self.computegrad(self.x_mat, self.y_array, theta_val, self.lambda_val)
            iter_v += 1
            obj_print = self.objective_func(self.x_mat, self.y_array, self.beta_val, self.lambda_val) 
            if (iter_v % 5 == 0):
                print "The objective value at iteration", iter_v, "is", round(obj_print, 10)
            
        if (iter_v == self.max_iter):
            print "Maximum number of iterations reached before convergence..."
        else:
            print "Algorithm has converged after", iter_v, "iterations!"
        
        # saving the final model beta value
        self.model_result = self.beta_val
    
    # writing the predict function
    def predict(self, x_pred):
        self.x_pred = x_pred
        rand = np.exp(self.x_pred.dot(self.model_result))
        final_r = rand/(rand+1)        
        final_r[final_r < 0.5] = -1
        final_r[final_r >= 0.5] = 1
        return final_r