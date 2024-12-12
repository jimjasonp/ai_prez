
def random_forest_reg(X_train,y,X_test):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()
    rf.fit(X_train,y)
    y_pred = rf.predict(X_test)
    return y_pred

def decision_tree_reg(X_train,y,X_test):
    from sklearn.tree import DecisionTreeRegressor
    dt = DecisionTreeRegressor()
    dt.fit(X_train,y)
    y_pred = dt.predict(X_test)
    return y_pred


def lasso_reg(X_train,y,X_test):
    from sklearn.linear_model import Lasso
    lasso = Lasso(
            alpha=0.000000002,
            max_iter=100000,
            selection='random'
            )
    lasso.fit(X_train,y)
    y_pred = lasso.predict(X_test)
    return y_pred

def svr(X_train,y,X_test):
    from sklearn.svm import SVR
    svr = SVR()
    svr.fit(X_train,y)
    y_pred = svr.predict(X_test)
    return y_pred


def gpr(X_train,y,X_test):
    from sklearn.gaussian_process import GaussianProcessRegressor
    gpr = GaussianProcessRegressor()
    gpr.fit(X_train,y)
    y_pred = gpr.predict(X_test)
    return y_pred



def elastic_net_cv(X_train,y,X_test):
    from sklearn.linear_model import ElasticNetCV
    encv = ElasticNetCV(
                cv= 50,
                eps=1e-6,
                l1_ratio=0.01,  
                max_iter=100000,  
                tol=1e-6,  
                selection='cyclic'
                )
    encv.fit(X_train,y)
    y_pred = encv.predict(X_test)
    return y_pred

def elastic_net(X_train,y,X_test):
    from sklearn.linear_model import ElasticNet
    en = ElasticNet(
                #alpha = 0.000000002,
                l1_ratio=0.01,
                alpha = 0.2,  
                max_iter=100000000,  
                tol=0.000001,  
                selection='random'
                )
    en.fit(X_train,y)
    y_pred = en.predict(X_test)
    return y_pred

def linear_regression(X_train,y,X_test):
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train,y)
    y_pred = lr.predict(X_test)
    return y_pred


def ridge_reg(X_train,y,X_test):
    from sklearn.linear_model import Ridge
    rr = Ridge(
        alpha=1,   
        max_iter=1000000, 
        tol=0.0000001, 
        solver='auto',  
        )
    rr.fit(X_train,y)
    y_pred = rr.predict(X_test)
    return y_pred

def mlp(X_train,y,X_test):

    import tensorflow as tf

    from tensorflow import keras

    from keras.models import Sequential

    from keras.layers import Flatten,Dense


    mlp = Sequential()
    # Flatten input from 28x28 images to 784 (28*28) vector
    #mlp.add(Flatten(input_shape=(None, 2250)))

    # Dense layer 1 (256 neurons)
    mlp.add(Dense(256, activation='sigmoid'))

    # Dense layer 2 (128 neurons)
    mlp.add(Dense(128, activation='sigmoid'))


    mlp.add(Dense(64, activation='sigmoid'))

    #mlp.add(Dense(32, activation='sigmoid'))
    # Output layer (10 classes)
    mlp.add(Dense(10, activation='sigmoid'))

    mlp.add(Dense(1, activation='linear'))


    mlp.compile(loss="mean_absolute_error", optimizer="adam")

    history = mlp.fit(X_train, y, epochs=150,verbose=0)
    y_pred = mlp.predict(X_test)
    return y_pred



def mlp_classifier(X_train,y,X_test):

    import tensorflow as tf

    from tensorflow import keras

    from keras.models import Sequential

    from keras.layers import Flatten,Dense


    mlp = Sequential()
    # Flatten input from 28x28 images to 784 (28*28) vector
    #mlp.add(Flatten(input_shape=(None, 2250)))

    # Dense layer 1 (256 neurons)
    mlp.add(Dense(256, activation='sigmoid'))

    # Dense layer 2 (128 neurons)
    mlp.add(Dense(128, activation='sigmoid'))


    mlp.add(Dense(64, activation='sigmoid'))

    #mlp.add(Dense(32, activation='sigmoid'))
    # Output layer (10 classes)


    mlp.add(Dense(3, activation='sigmoid'))

    #mlp.add(Dense(1, activation='linear'))


    mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = mlp.fit(X_train, y, epochs=150,verbose=0)
    y_pred = mlp.predict(X_test)
    return y_pred