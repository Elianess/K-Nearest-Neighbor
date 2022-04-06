#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
"""
Credits: the original code belongs to Stanford CS231n course assignment1. Source link: http://cs231n.github.io/assignments2019/assignment1/
"""

class KNearestNeighbor: # создаем класс - кнн
    """ a kNN classifier with L2 distance """
    # классификатор кнн с L2 дистанцией
    
    def __init__(self): # конструктор класса, можем вызвать метод конструктора для создания объекта
        pass # заглушка

    # методы описывают, что может делать класс
    def fit(self, X, y): # метод обучает классификатор(кнн). Для кнн это протсо запоминание обучающих данных
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        # принимает:
        # X - массив numpy формы (num_train, D) содержащий обучающие данные состоящие из образцов num_train для каждой размерности D
        # y - массив numpy формы (N,) содержащий обучающие метки, где y[i] - метка для X[i]
        
        self.X_train = X # создается атрибут X_train в который помещается принимаемый методом аргумент X 
        self.y_train = y # создается атрибут y_train в который помещается принимаемый методом аргумент y

    def predict(self, X, k=1, num_loops=0): # метод пердсказывает метки для тестовых данных используя классификатор(кнн)
        """
        Predict labels for test data using this classifier.
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        #принимает:
        # X - массив numpy формы (num_test, D) содержащий тестовые данные состоящие из образцов num_test для каждой размерности D
        # k - количество ближайших соседей, которые голосуют за предсказанные метки 
        # num_loops - определяет какую реализацю использовать, чтобы вычислить расстояние между обучающими и тестовыми точками
        #возвращает:
        # y - массив numpy формы (num_test,) содержащий предсказанные метки для тестовых данных, где y[i] - это предсказанная метка для тестовой точки X[i]
        
        if num_loops == 0: # если 0
            dists = self.compute_distances_no_loops(X) # то в переменную дистанции записывается результат метода с алгоритмом, где no loops 
        elif num_loops == 1: # если 1
            dists = self.compute_distances_one_loop(X) # то в переменную дистанции записывается результат метода с алгоритмом, где one loop
        elif num_loops == 2: # если 2
            dists = self.compute_distances_two_loops(X) # то в переменную дистанции записывается результат метода с алгоритмом, где two loops
        else: #иначе
            raise ValueError('Invalid value %d for num_loops' % num_loops) #вызывается исключение (недопустимое значение __ для num_loops)

        return self.predict_labels(dists, k=k) # возвращает результат метода predict_labels

    def compute_distances_two_loops(self, X): # метод вычисляет расстояние между тестовыми и обучающими точками с двумя циклами
        """кажд
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.
        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        #вычислите дистанцию между каждой тестовой точкой в X и каждой обучающей точкой в self.X_train используя вложенный цикл как по обучающим так и по тестовым данным 
        #принимает:
        # X - массив numpy формы (num_test, D) содержащий тестовые данные
        #возвращает:
        # dists - массив numpy формы (num_test, num_train) где dists[i, j] - Евклидово расстояние между i-той тестовой точкой и j-той обучающей точкой
        
        num_test = X.shape[0] # теперь содержит количество элементов массива с тестовыми данными
        num_train = self.X_train.shape[0] # теперь содержит количесвто элементов массива содержащего обучающие данные
        dists = np.zeros((num_test, num_train)) # создает массив из нулей где num_test - кол-во строк, а num_train - кол-во столбцов
        for i in range(num_test): # num_test раз будет выполняться цикл
            for j in range(num_train): # каждый i элемент num_test будет выполняться num_train раз
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                
                        #Сделать:
                        # вычислитm l2 расстояние между i тестовой точкой и j обучающей точкой и сохраните результат
                        # в dists[i, j]. Вам следует не использовать a loop over dimension и np.linalg.norm()
                
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j]))) #евклидово расстояние между 2-мя точками (в нашем случае между каждой тестовой точкой в X и каждой обучающей точкой в self.X_train)
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists #см в начало метода

    def compute_distances_one_loop(self, X): # метод вычисляет расстояние между тестовыми и обучающими точками с одним циклом
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.
        Input / Output: Same as compute_distances_two_loops
        """
        #вычислите расстояние между каждой тестовой точкой в X и каждой обучающей точкой в self.X_train используя одииночный цикл над тестовыми данными
        #принимает и возвращает то же что и compute_distances_two_loops
        
        num_test = X.shape[0] # теперь содержит количество элементов массива с тестовыми данными
        num_train = self.X_train.shape[0] # теперь содержит количесвто элементов массива содержащего обучающие данные
        dists = np.zeros((num_test, num_train)) # создает массив из нулей где num_test - кол-во строк, а num_train - кол-во столбцов
        for i in range(num_test): # num_test раз будет выполняться цикл
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            
                    #Сделать:
                    #вычислить l2 расстояние между i-той тестовой точкой и всеми обучающими точками
                    #сохраните результат в dists[i, :]. Не используйте np.linalg.norm()
                    
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            dists[i,:] = np.sqrt(np.sum(np.square(X[i]-self.X_train), axis=1)) #расстояния между каждой точкой тестовых данных и всеми обучающими
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists #см в начало метода

    def compute_distances_no_loops(self, X): # метод вычисляет расстояние между тестовыми и обучающими точками без циклов
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.
        Input / Output: Same as compute_distances_two_loops
        """
        #вычислите расстояние между каждой тествой точкой в X и каждой обучающей точкой в self.X_train без циклов 
        #принимает и возвращает то же что и compute_distances_two_loops
        
        num_test = X.shape[0] # теперь содержит количество элементов массива с тестовыми данными
        num_train = self.X_train.shape[0] # теперь содержит количесвто элементов массива содержащего обучающие данные
        dists = np.zeros((num_test, num_train)) # создает массив из нулей где num_test - кол-во строк, а num_train - кол-во столбцов
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        
                #Сделать:
                #Вычислить l2 расстояние между всеми тестовыми точками и всеми обучающими точками без циклов
                #сохранить результат в dists
                #Вы должны реализовать эту функцию используя только базовые операции массивов
                #в частности, вы не должны использовать функции из scipy, не используйте np.linalg.norm()
                #Подсказка:
                #Попробуйте сформулировать l2 расстояние используя матричное умножение и две широковещательные суммы
                
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dists = np.sqrt(np.sum(np.square(X[:,np.newaxis]-self.X_train[np.newaxis,:]), axis=2)) # https://numpy.org/doc/stable/user/basics.broadcasting.html
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists #см в начало метода

    def predict_labels(self, dists, k=1): # метод предсказывает метку для каждой тестовой точки
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        #Учитывая матрицу расстояний между тестовыми точками и обучающими точками, предсказать метки для каждой тестовой точки
        #Принимает: 
        # dists - массив numpy формы (num_test, num_train) где dists[i, j] дает растояние между i-той тестовой и j-той обучающей точками
        #Возвращает: 
        # y - массив numpy формы (num_test,) содержащий предказанные метки для тестовых данных, где y[i] предсказанныая метка для тестовой точки X[i]
        
        num_test = dists.shape[0] # теперь содержит количество элементов массива с тестовыми данными
        y_pred = np.zeros(num_test) # создает массив из нулей, где есть одна строка и кол-во элементов равно num_test
        for i in range(num_test): #num_test раз будет выполнятсья цикл
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
                    #список длинны k, хранит метки кнн к i-той тестовой точке 
            
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            
                    #Сделать:
                    #используйте матрицу расстояний, чтобы найти кнн i-той тестируемой точки и
                    #используйте self.y_train, чтобы найтти метки этих соседей. Сохраните эти метки в closest_y
                    #Подсказка:
                    #найдите функцию numpy.argsort
            
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            closest_y = []
            argsort = np.argsort(dists[i]) #индексы, сортирующие элементы исходного массива
            closest_y = self.y_train[argsort[:k]] #сохраняем подходящие метки
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            
                    #Сделать:
                    #Теперь, когда вы нашли метки кнн, вы должны найти самую распространенную метку
                    #в списке меток closest_y. Сохрание эти метки в y_pred[i]. Прекратите связь и выберите меньшую метку. 
            
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            (values, counts) = np.unique(closest_y, return_counts=True)
            y_pred[i] = values[np.argmax(counts)]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred

