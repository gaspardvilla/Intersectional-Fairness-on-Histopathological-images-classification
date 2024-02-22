# All categories we have in the data set for genders and races
GENDERS = {'str' : ['FEMALE', 'MALE'],
           'class' : [0, 1]}
RACES = {'str' : ['AMERICAN INDIAN OR ALASKA NATIVE', 'WHITE', 'ASIAN', 'BLACK OR AFRICAN AMERICAN'],
         'class' : [0, 1, 2, 3]}
AGES = {'str' : list(map(str, range(10))),
        'class' : list(range(10))}


# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #


METRICS = ['ACC', 'F1-score', 'TP', 'TN', 'FP', 'FN',
           'AUROC', 'AUC',
           'TPR - Recall', 'PPV - Precision', 'TNR', 'FPR', 'FNR', 'FDR', 'FOR', 'NPV', 'RPP', 'RNP',
           'SPD', 'DI', 'EOD', 'AAOD']


# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
        
        
# All possible runs parameters
if False:
    tasks = ['tumor_detection']#['cancer_classification', 'tumor_detection']
    cancer_dict = {'cancer_classification' : [#'coad_read_FS', 'coad_read_PM', 
                                            'kich_kirc_FS', #'kich_kirc_PM',
                                            'kich_kirp_PM', #'kich_kirp_FS', 'kich_kirp_PM',
                                            'kirc_kirp_FS', 'kirc_kirp_PM'],
                                            #'luad_lusc_FS', 'luad_lusc_PM'],
                # 'tumor_detection' : ['brca', 'coad', 'kich', 'kirc',
                #                         'kirp', 'luad', 'lusc', 'read'],
                # 'tumor_detection' : ['brca', 'luad', 'kirc', 'coad', 'lusc']
                'tumor_detection' : ['kirc']}
else:
    tasks = ['cancer_classification', 'tumor_detection']
    cancer_dict = {'cancer_classification' : ['coad_read_FS', 'coad_read_PM', 
                                              'kich_kirc_FS', 'kich_kirc_PM',
                                              'kich_kirp_FS', 'kich_kirp_PM',
                                              'kirc_kirp_FS', 'kirc_kirp_PM',
                                              'luad_lusc_FS', 'luad_lusc_PM'],
                   'tumor_detection' : ['brca', 'coad', 'kich', 'kirc',
                                        'kirp', 'luad', 'lusc', 'read']}
lambdas = [0.001, 0.005, 0.01, 0.05, 0.1]
pt_methods = ['DF_pos', 'DF_sum', 'DF_max']
alphas = [0.5]

# All the combinations for the run of the Foulds method
COMBS_FOULDS = []
for task in tasks:
    for cancer in cancer_dict[task]:
        for l in lambdas:
            for pt_method in pt_methods:
                COMBS_FOULDS += [[task, cancer, l, pt_method]]
                    
# All combinations for the baseline run
COMBS_BASELINE = []
for task in tasks:
    for cancer in cancer_dict[task]:
        COMBS_BASELINE += [[task, cancer]]
        
# All combinations for the Martinez run
COMBS_MARTINEZ = []
for task in tasks:
    for cancer in cancer_dict[task]:
        for a in alphas:
            COMBS_MARTINEZ += [[task, cancer, a]]