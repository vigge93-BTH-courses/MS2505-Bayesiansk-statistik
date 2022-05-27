from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB


def print_LaTeX(models, results):
    """Create a latex table of the results."""
    print('\\begin{table}[H]')
    print('\\begin{tabular}{|' + 'c|'*(2*len(models) + 1) + '}')
    print('\\hline & \\multicolumn{' + str(2*len(models)) + '}{c|}{Model} \\\\ \\hline ')

    for model in models:
        print(' & \\multicolumn{2}{c|}{' + model + '}', end=' ')
    print('\\\\ \\hline')

    print('Test size' + ' & Incorrect & Accuracy'*len(models) + '\\\\ \\hline')

    for test_size in results:
        print(f'{test_size*10}\\%', end='')

        for model in models:
            value = results[test_size][model]
            print('', value[2], value[3], sep=' & ', end='')

        print(' \\\\ \\hline')

    print('\\end{tabular}')
    print('\\caption{Result from the different training/test sizes and models}')
    print('\\label{tab:result}')
    print('\\end{table}')


def print_statistics(model, test_size, no_points, incorrect, accuracy):
    """"Print the statistics of the model."""
    print('='*20)
    print(f'Model: {model}')
    print(f'Testing size: {test_size*10}%')
    print(f'Total number of points: {no_points}')
    print(f'Incorrect points: {incorrect}')
    print(f'Accuracy: {accuracy*100}%')
    print('='*20)
    print()


X, y = load_iris(return_X_y=True)

gaussian_nb = GaussianNB()
categorical_nb = CategoricalNB()
bernoulli_nb = BernoulliNB()
models = {'Gaussian': gaussian_nb, 'Categorial': categorical_nb, 'Bernoulli': bernoulli_nb}

results = {}

for model in models:
    for test_size in range(1, 9):
        X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=test_size/10, random_state=0)

        y_pred_model = models[model].fit(X_training, y_training).predict(X_test)

        incorrect_predictions = (y_test != y_pred_model).sum()
        accuracy = 1-incorrect_predictions/X_test.shape[0]

        if test_size not in results:
            results[test_size] = {}
        results[test_size][model] = ((model, f'{test_size*10}\\%', incorrect_predictions, f'{accuracy*100:.2f}\\%'))

        print_statistics(model, test_size, X_test.shape[0], incorrect_predictions, accuracy)

print_LaTeX(models, results)
