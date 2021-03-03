

# create test/train files
## split shuffled dataset into training(80%) and testing(20%)
def create_sets(dataset, lower, upper, shuffle_indices):
    with open(dataset, 'a') as f:
        for i in range(lower, upper):
            line = str(shuffle_indices[i])
            f.write(line + '\n')
            print(line, end="\r", flush=True)
    print('saved...', dataset)