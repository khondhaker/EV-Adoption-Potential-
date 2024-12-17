import pandas as pd
import csv
import time
from tqdm import tqdm
import openai

# Function to construct the prompt
def construct_prompt(tweet):
    categories = {
        1: "Charging Infrastructure: EV charging stations, fast vs. home chargers, charging times, costs, and new technologies.",
        2: "Vehicle Features and Performance: Battery range, speed, safety, self-driving capabilities, and advanced technologies.",
        3: "Brands and Models: Tesla, Rivian, Ford, or other EV brands and models.",
        4: "Environmental Impact: Reducing carbon emissions, renewable energy integration, and mining concerns.",
        5: "Policies and Incentives: EV subsidies, tax credits, mandates, and regulations.",
        6: "Cost and Consumer Experience: EV pricing, maintenance savings, insurance rates, and user reviews.",
        7: "Industry Trends and News: EV market growth, announcements, investments, and partnerships.",
        8: "Challenges and Criticism: Infrastructure gaps, high costs, battery concerns, and EV adoption challenges.",
        0: "Irrelevant: Tweets that do not fit any category."
    }
    prompt = "Classify the following tweet into one of these categories:\n\n"
    for category, description in categories.items():
        prompt += f"{category}: {description}\n"
    prompt += f"\nTweet: {tweet}\n\nProvide only the category number (1-8) or '0' for irrelevant."
    return prompt

# Function to classify a single tweet
def classify_tweet(tweet):
    try:
        messages = [
            {"role": "system", "content": "You are an expert in classifying tweets about electric vehicles (EVs)."},
            {"role": "user", "content": construct_prompt(tweet)}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=10,
            temperature=0.0
        )
        classification = response['choices'][0]['message']['content'].strip()
        return classification
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Sleeping for 60 seconds...")
        time.sleep(60)  # Retry after 60 seconds
        return "Error"
    except openai.error.OpenAIError as e:
        if "quota_exceeded" in str(e):
            print(f"Daily quota exceeded at {time.strftime('%Y-%m-%d %H:%M:%S')}. Sleeping for 24 hours...")
            time.sleep(86400)  # Sleep for 24 hours
        return "Error"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "Error"

# Main function
def classify_tweets_auto_resume(input_csv, output_csv, daily_limit=10_000):
    total_requests = 0
    data = pd.read_csv(input_csv)

    # Load already processed tweets
    try:
        processed = pd.read_csv(output_csv)["Tweet_ID"].tolist()
    except FileNotFoundError:
        processed = []

    # Track processed IDs in memory
    processed_ids = set(processed)

    # Filter unprocessed data
    unprocessed_data = data[~data["Tweet_ID"].isin(processed_ids)]

    # Write to CSV with append mode
    with open(output_csv, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Tweet_ID", "cleaned_tweet", "classification"])

        while not unprocessed_data.empty:
            request_count = 0  # Reset daily counter
            for _, row in tqdm(unprocessed_data.iterrows(), total=len(unprocessed_data), desc="Classifying Tweets"):
                if request_count >= daily_limit:
                    print(f"Daily limit of {daily_limit} reached at {time.strftime('%Y-%m-%d %H:%M:%S')}. Sleeping for 24 hours...")
                    time.sleep(86400)  # Sleep for 24 hours
                    break

                tweet_id = row["Tweet_ID"]
                cleaned_tweet = row["cleaned_tweet"]

                if tweet_id in processed_ids:
                    continue  # Skip already processed IDs

                if not isinstance(cleaned_tweet, str) or cleaned_tweet.strip() == "":
                    writer.writerow([tweet_id, cleaned_tweet, "Error"])
                else:
                    classification = classify_tweet(cleaned_tweet)
                    writer.writerow([tweet_id, cleaned_tweet, classification])

                f.flush()  # Save progress immediately
                processed_ids.add(tweet_id)
                request_count += 1
                total_requests += 1

            print(f"Processed {request_count} tweets. Total processed: {total_requests}")
            # Re-filter unprocessed data after each batch
            unprocessed_data = data[~data["Tweet_ID"].isin(processed_ids)]

    print("All tweets have been processed successfully.")

# Example usage
input_csv = "Co_Tweets_unclassified.csv"
output_csv = "LLM_Classified_Tweets_gpt4o_mini.csv"
classify_tweets_auto_resume(input_csv, output_csv)
