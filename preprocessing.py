import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.decomposition import PCA

"""
Preprocessing of training data.
"""

def recodeMissing(df):
    """
    Recodes the missing values specified by "9", "99", "999", or "9999".
    
    @param df The unprocessed dataframe.
    
    @returns df The recoded dataframe.
    
    """
    cols9 = ["flag_fthb", "occpy_sts", "channel", "loan_purpose", "prop_val_meth", "pgrm_ind"]
    df[cols9] = df[cols9].replace(["9", 9], np.nan)
    
    cols99 = ["cnt_units", "prop_type", "cnt_borr"]
    df[cols99] = df[cols99].replace(["99", 99], np.nan)
    
    cols999 = ["mi_pct", "cltv", "dti", "ltv"]
    df[cols999] = df[cols999].replace(["999", 999], np.nan)
    
    cols9999 = ["fico"]
    df[cols9999] = df[cols9999].replace(["9999", 9999], np.nan)
    
    return df

def recodeZipcode(df):
    """
    Recodes the zipcode column into three factors using PCA.
    
    @param df The unprocessed dataframe.
    
    @returns df The recoded dataframe.
    
    """
    df.zipcode = df['zipcode'].astype('category')

    zipcode = df[['zipcode']]

    encoder_zip = ce.BackwardDifferenceEncoder(cols=['zipcode'])

    df_zip = encoder_zip.fit_transform(zipcode)
    
    pca_df_zip = PCA(n_components=3).fit_transform(df_zip)

    e_dataframe = pd.DataFrame(pca_df_zip,columns=['zip_1', 'zip_2', 'zip_3'],index=df.index)
    
    df[['zip_1', 'zip_2','zip_3']] = e_dataframe
    
    return df

def recodeDummies(df):
    df.cnt_units = df.cnt_units.map(dict({1: 1, 2: "more", 3: "more", 4: "more"}))
    
    dummy_cols = ["flag_fthb", "flag_sc", "int_only_ind", "rel_ref_ind"]
    
    df[dummy_cols] = df[dummy_cols].apply(lambda x: x.map(dict(Y=1, N=0)))
    
    # flag_sc: whitespace = Not Super Conforming --> recode NaN to 0
    df["flag_sc"] = df["flag_sc"].fillna(0)
    
    # rel_ref_ind: whitespace = Non-Relief Refinance loan
    df["rel_ref_ind"] = df["rel_ref_ind"].fillna(0)
    
    return df

def getDummies(df):
    cat_vars = ["cnt_units", "occpy_sts", "channel", "loan_purpose", "prop_type"]
    
    df = pd.get_dummies(df,columns=cat_vars,drop_first=False)
    
    return df

def removeColumns(df):
    vars_to_remove = ["seller_name", 
                  "servicer_name",
                  "pre_relief",
                  "prop_val_meth",
                  "st",
                  "pgrm_ind", 
                  "cd_msa", 
                  "zipcode",
                  "prod_type",
                  "ppmt_pnlty",
                  "dt_first_pi",
                  "dt_matr",
                  "id_loan"
                 ]
    
    df = df.drop(vars_to_remove, 1)
    
    return df
    
def normalizeFeatures(df):
    cont_vars = ["fico", "mi_pct", "cltv", "dti", "orig_upb", "ltv", "int_rt", "orig_loan_term", "cnt_borr"]
    
    std_dev = np.std(df[cont_vars], 0)
    std_dev[std_dev == 0] = 1
    mean_df = np.mean(df[cont_vars], 0)

    df[cont_vars] = (df[cont_vars] - np.full(df[cont_vars].shape, mean_df)) / np.full(df[cont_vars].shape, std_dev)

    return df

