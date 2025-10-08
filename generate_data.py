"""
Activity 11: Final Project - Generate Facebook Post Data
Student: Alvhin C. Solo
Course: Intelligent Systems in Forensics
"""

import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

LEGITIMATE_POSTS = [
    "Just finished reading an amazing book! Highly recommend it.",
    "Beautiful sunset at the beach today. Nature is incredible.",
    "Congratulations to our team for winning the championship!",
    "Happy birthday to my best friend! Hope you have a wonderful day.",
    "Check out this recipe I tried today. It turned out delicious!",
    "Excited to announce my new job at ABC Company!",
    "Family gathering this weekend was so much fun.",
    "Just adopted a cute puppy. Meet Max!",
    "Loving the new coffee shop downtown. Great atmosphere.",
    "Finished my first marathon today! Feeling accomplished.",
    "Great concert last night. The band was fantastic.",
    "Just planted some vegetables in my garden.",
    "Movie night with friends. Perfect way to end the week.",
    "Celebrating our anniversary today. Love you honey!",
    "Morning workout complete. Ready to start the day!",
    "Proud of my little sister for graduating with honors!",
    "Amazing day volunteering at the local shelter.",
    "Tried a new restaurant today and the food was incredible!",
    "So grateful for the support of my friends and family.",
    "Watching the stars tonight — feeling peaceful and blessed.",
    "New blog post is up! Sharing my travel experiences from Japan.",
    "Weekend hike was breathtaking. Can’t wait to go back!",
    "Learning to play the guitar — progress feels great!",
    "Supporting local businesses is always a good idea.",
    "Just finished a coding project I’ve been working on for weeks!"
]

MISINFORMATION_POSTS = [
    "BREAKING! Scientists discover cure for all diseases! Government hiding it from you!",
    "SHOCKING truth about vaccines that doctors don't want you to know!",
    "You won't believe what celebrities are secretly doing! Click now!",
    "URGENT! This miracle remedy will change your life forever!",
    "Secret government plan EXPOSED! Share before it's deleted!",
    "BANNED by mainstream media! The truth they don't want you to see!",
    "Amazing discovery that pharmaceutical companies are hiding!",
    "This one weird trick will solve all your problems instantly!",
    "BREAKING NEWS: Celebrity died! (actually fake) Share now!",
    "Doctors HATE this simple method! Click to learn more!",
    "Government conspiracy REVEALED! Wake up people!",
    "Miracle weight loss secret banned in 5 countries!",
    "SHOCKING evidence of alien cover-up! Must see!",
    "This natural cure DESTROYS cancer! Big pharma doesn't want you to know!",
    "URGENT WARNING! Something terrible happening right now!",
    "Bill Gates caught on camera spreading viruses — share this before it’s removed!",
    "NASA admits the moon landing was faked all along!",
    "Drinking lemon water can instantly cure COVID-19!",
    "World leaders secretly meeting to control the weather!",
    "Hidden message found in the new company logo — proof of global control!",
    "Doctors confirm: chocolate can extend your life by 50 years!",
    "The government is replacing birds with surveillance drones!",
    "Aliens have been living among us since 1947 — scientists confirm!",
    "Scientists admit the Earth is actually flat!",
    "Cure for diabetes found but kept secret for profit!"
]

SUSPICIOUS_DOMAINS = [
    "fakenews-source.net",
    "clickbait-central.com",
    "conspiracy-truth.org",
    "miracle-cures.net",
    "shocking-news-daily.com",
    "banned-truth.info",
    "secret-revelations.org",
    "viral-hoax.net",
    "truth-exposed.today",
    "hiddenfactsworld.com",
    "the-real-insider.net",
    "beforetheydelete.com",
    "fakehealthtips.org",
    "deepstateupdate.info",
    "worldalert24.net"
]

LEGITIMATE_DOMAINS = [
    "nytimes.com",
    "bbc.com",
    "wikipedia.org",
    "national-geographic.com",
    "scientificamerican.com",
    "reuters.com",
    "nature.com",
    "theguardian.com",
    "cnn.com",
    "forbes.com",
    "bloomberg.com",
    "nasa.gov",
    "who.int",
    "un.org",
    "time.com"
]

def generate_random_timestamp():
    start_date = datetime.now() - timedelta(days=30)
    random_days = random.randint(0, 30)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    random_seconds = random.randint(0, 59)
    timestamp = start_date + timedelta(days=random_days, hours=random_hours,
                                       minutes=random_minutes, seconds=random_seconds)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def generate_facebook_posts(num_posts=300, misinfo_ratio=0.15):
    posts = []
    num_misinfo = int(num_posts * misinfo_ratio)
    num_legit = num_posts - num_misinfo

    # --- Legitimate Posts ---
    for i in range(num_legit):
        post_content = random.choice(LEGITIMATE_POSTS)
        if random.random() > 0.7:
            post_content += " #blessed #grateful"
        post = {
            'post_id': f"POST_{i+1:04d}",
            'user_id': f"user_{random.randint(1, 150):03d}",
            'timestamp': generate_random_timestamp(),
            'post_content': post_content,
            'num_shares': random.randint(0, 50),
            'num_reactions': random.randint(5, 100),
            'source_link': random.choice(LEGITIMATE_DOMAINS) if random.random() > 0.6 else "UNKNOWN",
            'sentiment_score': round(random.uniform(0.1, 0.9), 2),
            'flagged': 'False'
        }
        posts.append(post)

    # --- Misinformation Posts ---
    for i in range(num_misinfo):
        post_content = random.choice(MISINFORMATION_POSTS)
        if random.random() > 0.5:
            post_content = post_content.upper()
        post = {
            'post_id': f"POST_{num_legit + i + 1:04d}",
            'user_id': f"user_{random.randint(1, 150):03d}",
            'timestamp': generate_random_timestamp(),
            'post_content': post_content,
            'num_shares': random.randint(100, 1000),
            'num_reactions': random.randint(200, 800),
            'source_link': random.choice(SUSPICIOUS_DOMAINS) if random.random() > 0.3 else "UNKNOWN",
            'sentiment_score': round(random.uniform(-0.9, -0.2), 2),
            'flagged': 'True'
        }
        posts.append(post)

    random.shuffle(posts)
    df = pd.DataFrame(posts)
    assert len(df) == num_posts, f"Expected {num_posts} posts but got {len(df)}"
    return df

def main():
    OUTPUT_FILE = 'final_project_raw_data.csv'
    df = generate_facebook_posts(num_posts=300, misinfo_ratio=0.15)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Generated {len(df)} Facebook posts (saved to {OUTPUT_FILE})")

if __name__ == "__main__":
    main()
