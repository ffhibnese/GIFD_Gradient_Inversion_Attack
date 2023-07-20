# import nevergrad as ng

# def onemax(x):
#     return len(x) - x.count(1)

# # Discrete, ordered
# param = ng.p.TransitionChoice(range(7), repetitions=10)
# optimizer = ng.optimizers.DiscreteOnePlusOne(parametrization=param, budget=100, num_workers=1)
# print("Initial param:{}".format(param.value))
# recommendation = optimizer.provide_recommendation()
# for _ in range(optimizer.budget):
#     x = [optimizer.ask() for i in range(10)]
#     for item in x:
#         print("Before tell:{}".format(item.value))
#     loss = [onemax(i.value) for i in x]
#     for item in loss:
#         print("Before tell:{}".format(item))
#     break
#     # loss = onemax(*x.args, **x.kwargs)  # equivalent to x.value if not using Instrumentation
#     loss = onemax(x.value)
#     optimizer.tell(x, loss)
#     print("After tell:{}".format(optimizer.parametrization))

# recommendation = optimizer.provide_recommendation()
# print(recommendation.value)
import os
import pandas as pd
def save_to_table(out_dir, name, dryrun, **kwargs):
    """Save keys to .csv files. Function adapted from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_{name}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        if os.path.isfile(fname):
            old_table = pd.read_csv(fname)
        old_table = old_table.append(kwargs, ignore_index=True)
        # with open(fname, 'r') as f:
        #     reader = csv.reader(f, delimiter='\t')
        #     header = [line for line in reader][0]
    except Exception as e:
        print('Creating a new .csv table...')
        data = pd.DataFrame(columns = fieldnames)
        data = data.append(kwargs, ignore_index=True)
    if not dryrun:
        # Add row for this experiment
        data.to_csv(fname, index=None)

save_to_table('./','test', False,  model=[20, 20, 20],
                                        dataset='dataset')

# import pandas as pd
# import csv

# tmp_lst = []
# with open('experiments/ex5_deeperLayer/table_psnr.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         # print(row)
#         tmp_lst.append(row)
# df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0]) 
# print(df['layer0_psnr'])