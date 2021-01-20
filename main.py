from fuzzy_system.fuzzy_learning_helper import load_iris_data
from fuzzy_system.fuzzy_learning_system import FuzzyLearningSystem
import statistics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

for method in ('classical', 'general'):
    for evaluate in (0, 1):
        folds_result = []
        for fold in range(1, 11):
            X_train, y_train = load_iris_data(fold=fold, data_type='train')
            X_test, y_test = load_iris_data(fold=fold, data_type='test')

            learning_system = FuzzyLearningSystem(res=500)
            learning_system.fit(X_train, y_train, X_n=2, y_n=2, evaluate_rules=evaluate)
            score = learning_system.score(X_test, y_test, method=method)

            folds_result.append(score)

            print('method:', method, ' rating:', evaluate, 'fold:', fold, ' score:', score)

        print("mean: %s " % (statistics.mean(folds_result)))
        print("standard deviation: %s " % (statistics.stdev(folds_result)))


            #learning_system.analyse_rules()

            #df = pd.DataFrame()

            #for i in np.arange(0, 50, 1):

            #    y_hat = learning_system.get_result({'x': i})['Y']

           #     a_row = pd.Series([i, y_hat])
           #     row_df = pd.DataFrame([a_row])
           #     df = pd.concat([row_df, df])

            #plt.plot(X_train, y_train)
            #plt.plot(df[0], df[1])

            # plt.show()

            # learning_system.generate_rules_csv('sensor_rules.csv')

            #print(learning_system)
