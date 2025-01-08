import json

if __name__ == '__main__':
    with open('optimization_results.json', 'r') as f:
        data = json.load(f)
        data1 = data['data1']
        data2 = data['data2']
        soln1 = data1['optimal_solution']
        soln2 = data2['optimal_solution']
        print('Optimal solution for data1: {}'.format(soln1))