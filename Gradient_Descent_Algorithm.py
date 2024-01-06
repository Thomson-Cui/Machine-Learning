#!/usr/bin/env python
# coding: utf-8

# In[1]:


def Prediction(x, w, b):
    
    y = np.dot(x, w) + b
    
    return y

def Cost(x, y, w, b):
    
    m = x.shape[0]
    
    y_hat = Prediction(x, w, b)
    
    error = y_hat - y
    
    J_w_b = np.sum(error**2) / (2*m)
    
    return J_w_b

def Gradient_computing(x, y, w, b):
    
    m = x.shape[0]
    
    x_T = np.transpose(x)
    
    y_hat = Prediction(x, w, b)
    
    error = y_hat - y
    
    dj_dw = np.dot(x_T, error) / m
    
    dj_db = np.sum(error) / m
    
    return dj_dw, dj_db

def print_outcome(i, w, b, J_w_b):
    
    print("Itertaion =", i)
    
    print("w =", '[' + ', '.join(['{:.1e}'.format(num) for num in w]) + ']')
    
    print(f"b = {b:0.1e}") 
    
    print(f"J_w_b = {J_w_b:0.5e}")
    
    print("") 
    

def Gradient_descent(x, y, alpha, iterations):
    
    n = x.shape[1]
    
    w = np.zeros(n)
    
    b = 0
    
    hist = np.zeros(iterations)
    
    for i in range(iterations):
        
        dj_dw, dj_db = Gradient_computing(x, y, w, b)
        
        w = w - alpha * dj_dw
        
        b = b - alpha * dj_db
        
        J_w_b = Cost(x, y, w, b)
        
        hist[i] = J_w_b
        
        if iterations <= 10:
            
            print_outcome(i, w, b, J_w_b)
        
        if 10 < iterations <= 100:
        
            if i % 10 == 0:
        
                print_outcome(i, w, b, J_w_b)
            
        if 100 < iterations <= 1000:
            
            if i % 100 == 0:
        
                print_outcome(i, w, b, J_w_b) 
        
    if iterations != 10:
    
        print_outcome(i, w, b, J_w_b)
    
    return w, b, J_w_b, hist

