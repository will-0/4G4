iris = load_iris()
X = iris['data'] # array of samples 4 dimensions each describing a feature
y = iris['target'] # array of labels (0, 1, 2)
names = iris['target_names'] # array of labels (0, 1, 2)
feature_names = iris['feature_names'] # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# One hot encoding
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray() # Y is output of 3 dimensions now, one hot encoding

# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
Iris_Xtrain, Iris_Xtest, Iris_Ytrain, Iris_Ytest = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)
print("Xtrain shape = {}".format(Iris_Xtrain.shape))
print("Ytrain shape = {}".format(Iris_Ytrain.shape))
print("Xtest shape = {}".format(Iris_Xtest.shape))
print("Ytest shape = {}".format(Iris_Ytest.shape))

n_features = X.shape[1]
n_classes = Y.shape[1]