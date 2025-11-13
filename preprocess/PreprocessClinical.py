import os
import pandas as pd
import imblearn
import numpy as np
from imblearn.over_sampling import SMOTE


class PreprocessClinical:

    def pre_clinical(self, df_clinical):
        # Censoring 9 -> 0
        df_clinical.loc[(df_clinical['Deadstatus.event'] == 9, 'Deadstatus.event')] = 0
        df_clinical['Overall.stage'] = df_clinical['Overall.stage'].str.upper()
        # Overall.stage 4 and 7
        # Overall.stage 값 변경
        # categorical data
        col_list = df_clinical.columns.tolist()
        print('col_list: ', col_list)

        # Remove Mcode and Mcode.description
        # if 'Mcode' in col_list:
        #     df_clinical.drop(['Mcode'], axis='columns', inplace=True)
        if 'Mcode.description' in col_list:
            df_clinical.drop(['Mcode.description'], axis=1, inplace=True)

        df_clinical.drop(['Histology', 'Smoking.status', 'Smoking.amount'], axis=1, inplace=True)
        df_clinical_dummy = pd.get_dummies(df_clinical, columns=['gender', 'Overall.stage', 'Clinical.T.Stage',
                                                                 'Clinical.N.stage', 'Clinical.M.stage'])
        # print('df_clinical_dummy :', df_clinical_dummy)
        print(df_clinical_dummy.info())  # Check the final column of dataset
        output_path = '../output'

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        df_clinical_dummy.to_excel('../output/CLINICAL_{}_pre.xlsx'.format(len(df_clinical)), index=False)
        print('>> [MESSAGE] SUCCESS: CLINICAL 전처리 완료')
        return df_clinical_dummy
