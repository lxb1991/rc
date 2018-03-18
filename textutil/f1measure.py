from loader import semeval
import os
import subprocess


class F1Measure:

    SCRIPT = '../semeval08/f1/semeval2010_task8_scorer-v1.2.pl'
    RELATION_TYPES_PATH = '../semeval08/f1/relation_types.txt'
    REAL_RELATION = '../semeval08/f1/real_result.txt'
    PREDICT_RELATION = '../semeval08/f1/prediction_result.txt'

    def __init__(self, labels):
        self.id2relations = semeval.load_relation_type(self.RELATION_TYPES_PATH)
        if not os.path.exists(self.REAL_RELATION):
            print('real relation not exist ! create it')
            with open(self.REAL_RELATION, 'w') as file:
                for index in range(len(labels)):
                    file.write(str(index) + '\t' + self.id2relations[labels[index]])

    def f1_score(self, labels):
        """
            计算 f1 score
        """
        with open(self.PREDICT_RELATION, 'w') as file:
            for index in range(len(labels)):
                file.write(str(index) + '\t' + self.id2relations[labels[index]])
        output = subprocess.getoutput('perl ' + self.SCRIPT + ' ' + self.PREDICT_RELATION + ' ' + self.REAL_RELATION)
        f1 = float(output[-10:-5])
        return f1
