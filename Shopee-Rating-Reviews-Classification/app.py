import base64
import io
import json
import ast
import matplotlib.pyplot as plt
from nltk import word_tokenize
from wordcloud import WordCloud
import plotly
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from flask import Flask, request, render_template
from Python.data_collection import scraping
from Python.data_preprocessing import pre_processing, remove_stopwords
from Python.test_prediction import test_prediction
from Python.class_def import TextSelector, NumberSelector


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/track')
def track():
    return render_template("track_product.html")


df = []


@app.route('/scrape', methods=['POST', 'GET'])
def scrape():
    product_names_list = []
    if request.method == 'POST':
        url = request.form['url']
        product_index = f"Product {len(df)+1}"
        data = scraping(url, product_index)
        df.append(data)

        df_to_csv = pd.concat(df, ignore_index=True)
        df_to_csv.to_csv("input.csv", encoding='utf-8', index=False)

        products = df_to_csv['product_index'].unique()

        for prod in products:
            frame = df_to_csv[df_to_csv['product_index'] == prod]
            product_names = frame['product_name'].unique()
            for name in product_names:
                product_names_list.append(name)

        print(product_names_list)
        return render_template("product_list.html", product_list=product_names_list)


@app.route('/review', methods=['POST', 'GET'])
def review():
    row_list = []

    csv_df = pd.read_csv("input.csv", encoding="utf-8")
    csv_df['date_created'] = pd.to_datetime(csv_df['date_created'])
    csv_df['date_created'] = csv_df['date_created'].dt.strftime('%Y-%m-%d')

    products = csv_df['product_index'].unique()
    reviews = []
    product_names = []
    for prod in products:
        frame = csv_df[csv_df['product_index'] == prod]
        product_names.extend(frame['product_name'].unique())
        review_frame = frame[['date_created', 'username', 'rating', 'review']].values.tolist()
        reviews.append(review_frame)

    for r in reviews:
        row_count = 0
        for row in r:
            row_count += 1
        print(row_count)
        row_list.append(row_count)

    my_dict = enumerate(products)
    return render_template('rating_and_reviews.html', reviews=reviews, button_labels=products, enumerate_obj=my_dict,
                           products=product_names, row_list=row_list)


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    row_list = []
    #predict_df = pd.read_csv("input_predicted.csv", encoding='utf-8')
    csv_df = pd.read_csv("input.csv", encoding="utf-8")
    csv_df['date_created'] = pd.to_datetime(csv_df['date_created'])
    csv_df['date_created'] = csv_df['date_created'].dt.strftime('%Y-%m-%d')

    pre_processed_df = pre_processing(csv_df)
    predict_df = test_prediction(pre_processed_df)

    pre_processed_df.to_csv("input_preprocessed.csv", encoding='utf-8', index=False)
    predict_df.to_csv("input_predicted.csv", encoding='utf-8', index=False)

    products = predict_df['product_index'].unique()
    products = sorted(products)
    reviews = []
    product_names = []
    for prod in products:
        frame = predict_df[predict_df['product_index'] == prod]
        product_names.extend(frame['product_name'].unique())
        review_frame = frame[['date_created', 'username', 'rating', 'review', 'cleaned', 'filtered', 'finalized',
                              'sentiment_score', 'duplicated_spam', 'product_index', 'category']].values.tolist()
        reviews.append(review_frame)

    for r in reviews:
        row_count = 0
        for row in r:
            row_count += 1
        print(row_count)
        row_list.append(row_count)

    my_dict = enumerate(products)
    return render_template('classified_reviews.html', reviews=reviews, button_labels=products, enumerate_obj=my_dict,
                           products=product_names, row_list=row_list)


@app.route('/recommend')
def recommend():

    csv_df = pd.read_csv("input_predicted.csv", encoding="utf-8")

    products = csv_df['product_index'].unique()
    shop_names = []
    max_shop = ''
    max_product = ''
    max_rating = -float('inf')
    max_rate_num_useful_reviews = 0
    max_product_url = ''
    max_num_useful_reviews = 0

    for prod in products:
        frame = csv_df[csv_df['product_index'] == prod]
        shop_names.extend(frame['shop_name'].unique())
        mean_rating = frame['rating'][frame['category'] == 'useful'].mean()
        num_useful_reviews = frame.loc[frame['category'] == 'useful'].shape[0]
        num_reviews = frame.shape[0]
        shops = frame['shop_name'].unique()
        product = frame['product_index'].unique()
        product_url = frame['product_url'].unique()

        rate_num_useful_reviews = num_useful_reviews/num_reviews

        if (mean_rating > max_rating) & (rate_num_useful_reviews > max_rate_num_useful_reviews):
            max_rating = mean_rating
            max_shop = shops[0]
            max_rate_num_useful_reviews = rate_num_useful_reviews
            max_num_useful_reviews = num_useful_reviews
            max_num_reviews = num_reviews
            max_product = product[0]
            max_product_url = product_url[0]
            # max_last_review_date = last_review_date

    print("max", max_rating, max_rate_num_useful_reviews)
    return render_template('recommendation.html', product_url=max_product_url, recommend=max_shop, mean_rate=round(max_rating, 2), review_count=max_num_reviews, useful_review_count=max_num_useful_reviews, product=max_product)


@app.route('/rating_analysis')
def rating_analysis():

    csv_df = pd.read_csv("input_predicted.csv", encoding="utf-8")
    products = csv_df['product_index'].unique()
    products = sorted(products)
    useful_reviews = csv_df[csv_df['category'] == 'useful']

    # Trend of Monthly Average Ratings in 2023

    useful_reviews['date_created'] = pd.to_datetime(useful_reviews['date_created'])
    useful_reviews = useful_reviews.sort_values('date_created')

    useful_reviews['month_year'] = useful_reviews['date_created'].dt.to_period('M')
    monthly_avg_rating = useful_reviews.groupby(['month_year', 'shop_name'])['rating'].mean().reset_index()
    monthly_avg_rating_2023 = monthly_avg_rating[monthly_avg_rating['month_year'] >= '2023-01']
    monthly_avg_rating_2023['month_year'] = monthly_avg_rating['month_year'].astype(str)

    fig = px.line(monthly_avg_rating_2023, x='month_year', y='rating', color='shop_name', markers=True)

    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Average Rating', dtick=1)
    fig.update_layout(title='Monthly Average Rating Over Time', title_x=0.5, legend_title_text='Shop',
                      yaxis_range=[0, 6],
                      legend=dict(
                          yanchor='top',
                          y=-0.2,
                          xanchor='left',
                          x=0.01
                      ))

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Trend of Daily Average Ratings in 2023

    useful_reviews['date_created'] = pd.to_datetime(useful_reviews['date_created'])
    useful_reviews = useful_reviews.sort_values('date_created')

    daily_avg_rating = useful_reviews.groupby(['date_created', 'shop_name'])['rating'].mean().reset_index()
    daily_avg_rating_2023 = daily_avg_rating[daily_avg_rating['date_created'] >= '2023-01-01']
    daily_avg_rating_2023['date_created'] = daily_avg_rating_2023['date_created'].astype(str)

    fig1 = px.line(daily_avg_rating_2023, x='date_created', y='rating', color='shop_name', markers=True)
    fig1.update_xaxes(title_text='Date')
    fig1.update_yaxes(title_text='Average Rating', dtick=1)
    fig1.update_layout(title='Daily Average Rating Over Time', title_x=0.5, legend_title_text='Shop',
                       yaxis_range=[0, 6],
                       legend=dict(
                           yanchor='top',
                           y=-0.2,
                           xanchor='left',
                           x=0.01
                       ))

    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    # Trend of Monthly Average Ratings in 2022

    useful_reviews['date_created'] = pd.to_datetime(useful_reviews['date_created'])
    useful_reviews = useful_reviews.sort_values('date_created')

    useful_reviews['month_year'] = useful_reviews['date_created'].dt.to_period('M')
    monthly_avg_rating = useful_reviews.groupby(['month_year', 'shop_name'])['rating'].mean().reset_index()
    monthly_avg_rating_2022 = monthly_avg_rating[(monthly_avg_rating['month_year'] >= '2022-01') &
                                                 (monthly_avg_rating['month_year'] < '2023-01')]
    monthly_avg_rating_2022['month_year'] = monthly_avg_rating['month_year'].astype(str)

    fig2 = px.line(monthly_avg_rating_2022, x='month_year', y='rating', color='shop_name', markers=True)

    fig2.update_xaxes(title_text='Date')
    fig2.update_yaxes(title_text='Average Rating', dtick=1)
    fig2.update_layout(title='Monthly Average Rating Over Time', title_x=0.5, legend_title_text='Shop',
                       yaxis_range=[0, 6],
                       legend=dict(
                          yanchor='top',
                          y=-0.2,
                          xanchor='left',
                          x=0.01
                       ))

    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # Trend of Daily Average Ratings in 2022

    useful_reviews['date_created'] = pd.to_datetime(useful_reviews['date_created'])
    useful_reviews = useful_reviews.sort_values('date_created')

    daily_avg_rating = useful_reviews.groupby(['date_created', 'shop_name'])['rating'].mean().reset_index()
    daily_avg_rating_2022 = daily_avg_rating[(daily_avg_rating['date_created'] >= '2022-01-01') &
                                             (daily_avg_rating['date_created'] < '2023-01-01')]
    daily_avg_rating_2022['date_created'] = daily_avg_rating_2022['date_created'].astype(str)

    fig3 = px.line(daily_avg_rating_2022, x='date_created', y='rating', color='shop_name', markers=True)
    fig3.update_xaxes(title_text='Date')
    fig3.update_yaxes(title_text='Average Rating', dtick=1)
    fig3.update_layout(title='Daily Average Rating Over Time', title_x=0.5, legend_title_text='Shop',
                       yaxis_range=[0, 6],
                       legend=dict(
                           yanchor='top',
                           y=-0.2,
                           xanchor='left',
                           x=0.01
                       ))

    graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    # Comparing Rating Distribution for Each Product

    grouped_data = useful_reviews.groupby(['shop_name', 'rating']).size().unstack().reset_index()
    melted_data = pd.melt(grouped_data, id_vars='shop_name',
                          var_name='rating', value_name='count')

    fig4 = px.bar(melted_data, x='shop_name', y='count', color='rating',
                  labels={'shop_name': 'Shop', 'count': 'Count', 'rating': 'Rating'},
                  barmode='group')
    fig4.update_xaxes(title_text='Shop', tickangle=45)
    fig4.update_yaxes(title_text='Count')
    fig4.update_layout(title='Rating Distribution by Shop', title_x=0.5)
    fig4.update_layout(legend_title_text='Shop')

    graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    fig5 = px.bar(melted_data, x='shop_name', y='count', color='rating',
                  labels={'shop_name': 'Shop', 'count': 'Count', 'rating': 'Rating'},
                  barmode='stack')
    fig5.update_xaxes(title_text='Shop', tickangle=45)
    fig5.update_yaxes(title_text='Count')
    fig5.update_layout(title='Rating Distribution by Shop', title_x=0.5)
    fig5.update_layout(legend_title_text='Shop')

    graphJSON5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

    # Comparing Average Rating Between Products Across Shops

    product_avg_rate = useful_reviews.groupby('shop_name').agg({'rating': 'mean', 'category': 'count'}).reset_index()
    product_avg_rate = product_avg_rate.rename(columns={'category': 'useful_reviews'})

    fig6 = px.scatter(product_avg_rate, x='shop_name', y='rating', labels={'shop_name': 'Shop', 'rating': 'Average Rating'},
                      title='Comparison of Average Rating Between Products')

    fig6.update_traces(mode='markers+lines')
    fig6.update_xaxes(tickangle=45)
    fig6.update_layout(
        xaxis_title='Shop',
        yaxis_title='Average Rating',
        showlegend=False,
        title_x=0.5,
        yaxis=dict(range=[product_avg_rate['rating'].min() - 0.5, product_avg_rate['rating'].max() + 0.5])
    )

    graphJSON6 = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

    fig7 = px.bar(product_avg_rate, x='shop_name', y='rating', labels={'shop_name': 'Shop', 'rating': 'Average Rating'},
                  title='Comparison of Average Rating Between Shops')

    fig7.update_layout(
        xaxis_title='Shop',
        yaxis_title='Average Rating',
        showlegend=False,
        title_x=0.5,
        yaxis=dict(range=[product_avg_rate['rating'].min() - 0.5, product_avg_rate['rating'].max() + 0.5])
    )

    fig7.update_xaxes(tickangle=45)
    graphJSON7 = json.dumps(fig7, cls=plotly.utils.PlotlyJSONEncoder)

    my_dict = enumerate(products)
    return render_template('rating_analysis.html', graphJSON=graphJSON, graphJSON1=graphJSON1, graphJSON2=graphJSON2,
                           graphJSON3=graphJSON3, graphJSON4=graphJSON4, graphJSON5=graphJSON5, graphJSON6=graphJSON6,
                           graphJSON7=graphJSON7, button_labels=products, enumerate_obj=my_dict)


def word_cloud(frame):
    frame['filtered'] = frame['filtered'].apply(lambda x: ast.literal_eval(x))

    unrelated_words = ['seller', 'delivery', 'service', 'penghantaran', 'bungkusan', 'package', 'parcel', 'servis', 'money', 'selamat', 'jadi']

    filtered_mentions = []
    for mention_list in frame['filtered']:
        for mention in mention_list:
            words = word_tokenize(mention)
            for word in words:
                if word not in unrelated_words:
                    filtered_mentions.append(word)

    text = " ".join(filtered_mentions)
    text = remove_stopwords(text)
    wordcloud = WordCloud(background_color="white").generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("Product Mentions")
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("utf-8")
    plt.close()

    return data


@app.route('/reviews_analysis')
def reviews_analysis():
    csv_df = pd.read_csv("input_predicted.csv", encoding="utf-8")
    products = csv_df['product_index'].unique()
    products = sorted(products)
    useful_reviews = csv_df[csv_df['category'] == 'useful']

    # Monthly Analysis on Reviews with Positive Rating in 2023

    useful_reviews['date_created'] = pd.to_datetime(useful_reviews['date_created'])
    useful_reviews = useful_reviews.sort_values('date_created')

    useful_reviews['month_year'] = useful_reviews['date_created'].dt.strftime('%Y-%m')

    monthly_num_reviews_above_3 = useful_reviews[useful_reviews['rating'] > 3].groupby(
        ['month_year', 'shop_name']).size().reset_index(name='num_reviews_above_3')
    monthly_num_reviews_above_3_2023 = monthly_num_reviews_above_3[
        monthly_num_reviews_above_3['month_year'] >= '2023-01']
    monthly_num_reviews_above_3_2023.loc[:, 'month_year'] = monthly_num_reviews_above_3_2023['month_year'].astype(str)

    fig = px.bar(monthly_num_reviews_above_3_2023, x='month_year', y='num_reviews_above_3', color='shop_name')
    fig.update_xaxes(title_text='Date', tickangle=45, dtick='M1')
    fig.update_yaxes(title_text='Number of Reviews', dtick=2)
    fig.update_layout(title="Reviews with Positive Rating", legend_traceorder="reversed",
                      title_x=0.2, legend_title_text='Shop', yaxis_range=[0, 20],
                       legend=dict(
                           yanchor='top',
                           y=-0.5,
                           xanchor='left',
                           x=0.01
                       ))

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Monthly Analysis on Reviews with Negative Rating in 2023

    monthly_num_reviews_below_3 = useful_reviews[useful_reviews['rating'] < 3].groupby(
        ['month_year', 'shop_name']).size().reset_index(name='num_reviews_below_3')

    monthly_num_reviews_below_3_2023 = monthly_num_reviews_below_3[
        monthly_num_reviews_below_3['month_year'] >= '2023-01']

    monthly_num_reviews_below_3_2023.loc[:, 'month_year'] = monthly_num_reviews_below_3_2023['month_year'].astype(str)

    fig1 = px.bar(monthly_num_reviews_below_3_2023, x='month_year', y='num_reviews_below_3', color='shop_name')
    fig1.update_xaxes(title_text='Date', tickangle=45, dtick='M1')
    fig1.update_yaxes(title_text='Number of Reviews', dtick=2)
    fig1.update_layout(title="Reviews with Negative Rating", legend_traceorder="reversed",
                       title_x=0.2, legend_title_text='Shop', yaxis_range=[0, 20],
                       legend=dict(
                           yanchor='top',
                           y=-0.5,
                           xanchor='left',
                           x=0.01
                       ))

    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    # Monthly Analysis on Reviews with Positive Rating in 2022

    monthly_num_reviews_above_3_2022 = monthly_num_reviews_above_3[
        (monthly_num_reviews_above_3['month_year'] >= '2022-01') & (
                    monthly_num_reviews_above_3['month_year'] < '2023-01')]

    monthly_num_reviews_above_3_2022.loc[:, 'month_year'] = monthly_num_reviews_above_3_2022['month_year'].astype(str)

    fig2 = px.bar(monthly_num_reviews_above_3_2022, x='month_year', y='num_reviews_above_3', color='shop_name')
    fig2.update_xaxes(title_text='Date', tickangle=45, dtick='M1')
    fig2.update_yaxes(title_text='Number of Reviews', dtick=2)
    fig2.update_layout(title="Reviews with Positive Rating", legend_traceorder="reversed",
                       title_x=0.2, legend_title_text='Shop', yaxis_range=[0, 20],
                       legend=dict(
                           yanchor='top',
                           y=-0.5,
                           xanchor='left',
                           x=0.01
                       ))

    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # Monthly Analysis on Reviews with Negative Rating in 2022

    monthly_num_reviews_below_3_2022 = monthly_num_reviews_below_3[
        (monthly_num_reviews_below_3['month_year'] >= '2022-01') & (
                    monthly_num_reviews_below_3['month_year'] < '2023-01')]

    monthly_num_reviews_below_3_2022.loc[:, 'month_year'] = monthly_num_reviews_below_3_2022['month_year'].astype(str)

    fig3 = px.bar(monthly_num_reviews_below_3_2022, x='month_year', y='num_reviews_below_3', color='shop_name')
    fig3.update_xaxes(title_text='Date', tickangle=45, dtick='M1')
    fig3.update_yaxes(title_text='Number of Reviews', dtick=2)
    fig3.update_layout(title="Reviews with Negative Rating", legend_traceorder="reversed",
                       title_x=0.2, legend_title_text='Shop', yaxis_range=[0, 20],
                       legend=dict(
                           yanchor='top',
                           y=-0.5,
                           xanchor='left',
                           x=0.01
                       ))

    graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    # Comparing Useful and Not Useful Reviews Across Shops

    grouped_data = csv_df.groupby(['shop_name', 'category']).size().unstack().reset_index()

    melted_data = pd.melt(grouped_data, id_vars='shop_name',
                          var_name='category', value_name='count')

    fig4 = px.bar(melted_data, x='shop_name', y='count', color='category',
                  barmode='group',
                  color_discrete_map={
                      'not useful': 'red',
                      'useful': 'rgb(47, 210, 150)'
                  })

    fig4.update_xaxes(title_text='Shop', tickangle=45)
    fig4.update_yaxes(title_text='Number of Reviews')
    fig4.update_layout(title="Comparison of Useful and Not Useful Reviews Across Shops", legend_traceorder="reversed",
                       title_x=0.2)

    graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    fig5 = px.bar(melted_data, x='shop_name', y='count', color='category',
                  barmode='stack', color_discrete_map={
                      'not useful': 'red',
                      'useful': 'rgb(47, 210, 150)'
                  })

    fig5.update_xaxes(title_text='Shop', tickangle=45)
    fig5.update_yaxes(title_text='Number of Reviews')
    fig5.update_layout(title="Comparison of Useful and Not Useful Reviews Across Shops", legend_traceorder="reversed",
                       title_x=0.2)

    graphJSON5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

    wordcloud_list = []
    for prod in products:
        frame = useful_reviews[useful_reviews['product_index'] == prod]
        wordcloud = word_cloud(frame)
        wordcloud_list.append(wordcloud)

    my_dict = enumerate(products)
    return render_template('reviews_analysis.html', graphJSON=graphJSON, graphJSON1=graphJSON1, graphJSON2=graphJSON2,
                           graphJSON3=graphJSON3, graphJSON4=graphJSON4, graphJSON5=graphJSON5, wordcloud=wordcloud_list,
                           enumerate_obj=my_dict)


if __name__ == "__main__":
    app.run()
