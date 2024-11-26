import pandas as pd
import pickle
from Python.class_def import TextSelector, NumberSelector


def test_prediction(df):
    with open('model/model_SVM_1.pkl', 'rb') as f:
        model = pickle.load(f)

    features = [c for c in df.columns.values if
                c not in ['username', 'review', 'cleaned', 'filtered', 'shop_name', 'product_name', 'date_created',
                          'product_index', 'product_url']]

    print(features)
    test_input = df[features]

    predict = model.predict(test_input)

    test_input = test_input.reset_index()

    df2 = pd.DataFrame(predict, columns=['category'])
    df_merged = pd.concat([test_input, df2], axis=1)

    df_merged['category'] = df_merged['category'].replace({0: 'not useful', 1: 'useful'})
    df_merged['duplicated_spam'] = df_merged['duplicated_spam'].replace({0: 'no match', 1: 'found match'})
    df_merged.drop(['index'], axis=1, inplace=True)
    predict_df = pd.concat([df['date_created'], df['username'], df['review'], df['shop_name'], df['product_name'],
                            df['product_index'], df['product_url'], df['cleaned'], df['filtered'], df_merged], axis=1)

    for col in df_merged.columns:
        print(col)

    return predict_df.sort_values(by=['category', 'date_created'], ascending=[False, False])



