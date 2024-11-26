import re
import requests
import pandas as pd
import datetime


def scrapeShopInfo(url):
    # Initialize header
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
    }

    # Retrieve shop id and item id
    r = re.search(r"i\.(\d+)\.(\d+)", url)

    # Initialize placeholder values
    shop_id, item_id = r[1], r[2]

    shop_url = "https://shopee.com.my/api/v4/product/get_shop_info?shopid={shop_id}"

    while True:
        data = requests.get(
            shop_url.format(shop_id=shop_id),
            headers=headers).json()

        if data["data"]["name"] is None:
            return "Sorry, shop name cannot be retrieved :("
            break

        else:
            shop_name = data["data"]["name"]
            break

    return shop_name


def scrapeProductReviews(url):

    # Initialize header
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
    }

    # Retrieve shop id and item id
    r = re.search(r"i\.(\d+)\.(\d+)", url)

    # Initialize placeholder values
    shop_id, item_id = r[1], r[2]
    review_limit = 20
    offset = 0
    review_count = 0

    # Retrieve ratings and reviews
    reviews_url = 'https://shopee.com.my/api/v2/item/get_ratings?filter=1&flag=1&limit={review_limit}&itemid={item_id}&offset={offset}&shopid={shop_id}&type=0'

    username = []
    star_rating = []
    comment = []
    date_created = []

    while True:
        data = requests.get(
            reviews_url.format(shop_id=shop_id, item_id=item_id, offset=offset, review_limit=review_limit),
            headers=headers, ).json()

        i = 1

        if data["data"]["ratings"] is None:
            break

        else:
            for i, rating in enumerate(data["data"]["ratings"], 1):
                date = datetime.date.fromtimestamp(rating["ctime"])
                # if datetime.date(2021, 1, 1) <= date <= datetime.date(2023, 12, 31):
                date_created.append(date)
                username.append(rating["author_username"])
                star_rating.append(rating["rating_star"])
                comment.append(rating["comment"])
                review_count += 1  # increment the counter variable

                if review_count > 100:  # break out of loop once review_count reaches limit
                    break
        if i % 20:
            break

        offset += 20

    # Save list into dataframe
    dfReview = pd.DataFrame(zip(date_created, username, star_rating, comment),
                            columns=['date_created', 'username', 'rating', 'review'])

    dfReview.dropna(how='any', inplace=True)

    return dfReview


def scrapeProductName(url):
    pattern = r'\/([^/]+)-i\.\d+\.\d+'

    # Search for the pattern in the URL
    match = re.search(pattern, url)

    if match:
        product_name = match.group(1).replace("-", " ")
        new_product_name = re.sub(r'%[^ ]*', '', product_name)

        return new_product_name


def scraping(url, product_index):

    review = scrapeProductReviews(url)
    product_name = scrapeProductName(url)
    product_names = [product_name] * len(review)
    shop_name = scrapeShopInfo(url)
    shop_names = [shop_name] * len(review)

    df1 = pd.DataFrame(product_names, columns=["product_name"])
    df2 = pd.DataFrame(shop_names, columns=["shop_name"])

    # Concatenating the new DataFrame with the existing DataFrame
    df = pd.concat([df1, df2, review], axis=1)

    df['product_index'] = product_index
    df['product_url'] = url

    df.dropna(how='any', inplace=True)

    return df



