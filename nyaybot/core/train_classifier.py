import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# ── Training Data ──────────────────────────────────────────────────────────────

theft_data = [
    # phone theft
    "my phone was stolen",
    "someone stole my mobile",
    "my smartphone was snatched",
    "phone stolen from my pocket",
    "my cell phone was taken by someone",
    "thief stole my phone at bus stop",
    "my phone got stolen in the market",
    "pickpocket took my mobile in crowd",
    "someone grabbed my phone and ran",
    "my phone was stolen while charging at cafe",
    # wallet/cash theft
    "my wallet was stolen",
    "someone pickpocketed my wallet",
    "cash stolen from my pocket",
    "my purse was snatched",
    "someone stole money from my bag",
    "my wallet was taken on the bus",
    "thief stole my purse at railway station",
    "my cash was stolen from office drawer",
    "pickpocket stole my wallet in metro",
    "someone took my wallet while I was sleeping",
    # vehicle theft
    "my bike was stolen",
    "my car was stolen from parking",
    "two wheeler stolen outside temple",
    "my scooter went missing from parking lot",
    "someone stole my bicycle",
    "my motorcycle was stolen last night",
    "vehicle stolen from outside my house",
    "my car was stolen from office parking",
    "auto rickshaw was stolen",
    "someone broke my car window and stole items",
    # house theft
    "my house was broken into",
    "robbery at my home",
    "thieves entered my house and stole valuables",
    "gold jewellery stolen from house",
    "my house was robbed while I was away",
    "burglars stole my TV and laptop",
    "someone broke into my flat",
    "my locker was broken and cash stolen",
    "thieves stole my documents and jewellery",
    "my house was robbed during night",
    # other theft
    "my laptop was stolen from office",
    "bag stolen at airport",
    "someone stole my gold chain",
    "my watch was snatched",
    "theft at my shop",
    "my documents were stolen",
    "someone stole goods from my shop",
    "my luggage was stolen at railway station",
    "someone stole my ATM card",
    "my camera was stolen at tourist spot",
]

scam_data = [
    # online scam
    "I was scammed online",
    "fake website took my money",
    "online fraud happened with me",
    "I lost money in internet fraud",
    "fake online store cheated me",
    "I paid for product online but never received",
    "fake shopping website stole my money",
    "I was cheated by online seller",
    "fraudulent website took my credit card details",
    "online marketplace seller cheated me",
    # phone/call scam
    "fake call saying I won lottery",
    "someone called pretending to be bank",
    "received fake KYC update call and lost money",
    "phone call fraud took my savings",
    "fake customer care call stole my OTP",
    "fraudster called as electricity department",
    "received call claiming to be from income tax",
    "fake insurance agent called and took money",
    "someone called as police and demanded money",
    "received threatening call asking for money",
    # UPI/banking fraud
    "UPI fraud happened with me",
    "someone transferred money from my account",
    "fake UPI payment request cheated me",
    "my bank account was emptied by fraud",
    "someone did unauthorized transaction from my account",
    "QR code scam took my money",
    "fake payment link stole my money",
    "someone sent me fake payment screenshot",
    "I was cheated in UPI collect request",
    "fraud transaction happened on my credit card",
    # job/investment scam
    "fake job offer took advance money",
    "investment fraud cheated me",
    "fake company took money promising job",
    "I lost money in ponzi scheme",
    "fake share market tips fraud",
    "multi level marketing scam cheated me",
    "fake mutual fund agent took my money",
    "property dealer fraud took my money",
    "fake visa agent cheated me for abroad job",
    "education loan fraud cheated me",
    # other scams
    "I was cheated by matrimony site",
    "fake NGO collected donation and disappeared",
    "received fake cheque that bounced",
    "astrologer fraud took my money",
    "fake medicine seller cheated me",
    "lottery fraud took my money",
    "chit fund company ran away with my money",
    "fake charity collected money",
    "someone cheated me in land deal",
    "advance fee fraud took my savings",
    # (money transfer focus)
    "I transferred money to fraud account",
    "fake person took money via gpay",
    "paid advance to fraud seller",
]

consumer_data = [
    # product issues
    "shop gave me defective product",
    "bought phone but it stopped working",
    "new appliance not working properly",
    "product quality is very poor",
    "I received damaged product",
    "item delivered was broken",
    "product not matching description",
    "fake product sold to me",
    "adulterated food sold at shop",
    "expired medicine sold at pharmacy",
    "new TV stopped working after 2 days",
    "washing machine stopped after 3 days",
    "refrigerator stopped cooling after one week",
    "laptop battery not working from day one",
    "air conditioner not cooling properly",

    # refund issues
    "company not giving refund",
    "shop refusing to return my money",
    "ecommerce site not processing refund",
    "I cancelled order but money not returned",
    "service provider not refunding advance",
    "gym not refunding my fees after closure",
    "travel agent not giving refund for cancelled trip",
    "hotel not refunding booking amount",
    "airline not giving refund for cancelled flight",
    "online course refund not processed",
    "cab app not refunding overcharge amount",
    "amazon not processing return request",
    "flipkart refusing refund on damaged item",

    # warranty/service issues
    "service center not repairing under warranty",
    "company refusing warranty claim",
    "authorized service center damaged my product",
    "warranty card not accepted",
    "company saying warranty is void without reason",
    "repair shop not returning my device",
    "service taking too long without update",
    "technician charged extra without informing",
    "repaired product broke again in one week",
    "service center lost my device",
    "refrigerator warranty rejected by company",
    "builder not completing construction work",

    # food/restaurant
    "restaurant served me stale food",
    "got food poisoning from restaurant",
    "foreign object found in food",
    "restaurant overcharged me",
    "restaurant not maintaining hygiene",
    "hotel gave me expired food",
    "food delivery app sent wrong order",
    "food delivered was completely different",
    "restaurant added extra charges to bill",
    "canteen serving unhygienic food",
    "swiggy delivered cold and stale food",
    "zomato order missing items",

    # billing/service issues
    "electricity bill is incorrect",
    "bank charged hidden fees",
    "broadband company not providing promised speed",
    "DTH service stopped without reason",
    "mobile recharge plan not activated properly",
    "internet service provider not giving refund",
    "credit card charged without my knowledge",
    "postpaid bill has wrong charges",

    # healthcare/other
    "hospital overcharged me for treatment",
    "insurance company rejected valid claim",
    "doctor gave wrong medicine",
    "amazon delivered empty package",
    "builder not giving flat possession",
    "builder using inferior construction material",
    "gym equipment broken but membership not paused",
    "cab overcharged and refuses refund",
]
cybercrime_data = [
    # account hacking
    "my social media account was hacked",
    "someone hacked my instagram",
    "my facebook account was taken over",
    "email account hacked by someone",
    "my twitter account was compromised",
    "hacker took control of my youtube channel",
    "my whatsapp was hacked",
    "someone logged into my account without permission",
    "my gmail was hacked and password changed",
    "my online banking account was hacked",
    "my snapchat account was hacked",
    "someone changed my instagram password",
    "my linkedin account was compromised",

    # image/video misuse
    "someone posted my private photos online",
    "morphed photos of me circulating on internet",
    "my private video leaked on whatsapp",
    "someone is sharing my intimate photos without consent",
    "fake photos of me posted on social media",
    "someone edited my photos and posted online",
    "my pictures misused to create fake profile",
    "obscene content created using my photos",
    "my video call was recorded without consent",
    "someone leaked my personal video online",
    "deepfake video created using my face",
    "my photos used to create fake dating profile",

    # identity/data theft online
    "fake profile created using my photos",
    "someone is impersonating me online",
    "my aadhaar details misused online",
    "identity theft using my personal information",
    "my data was leaked in breach",
    "someone applied for loan using my identity",
    "my PAN card details misused online",
    "fake account created on my name",
    "someone is using my name and photo on matrimony site",
    "my personal data sold on dark web",
    "someone filed fake tax return using my PAN",
    "my voter ID details being misused",

    # malware/ransomware
    "ransomware attacked my computer",
    "virus encrypted all my files",
    "my computer got hacked remotely",
    "malware installed on my laptop",
    "suspicious app stole my data",
    "my website was hacked",
    "company server hacked and data stolen",
    "received phishing link that stole my password",
    "spyware installed on my phone",
    "my camera was accessed remotely by hacker",
    "someone installed keylogger on my computer",
    "my phone was remotely accessed by hacker",
    "virus deleted all my important files",
    "fake website stole my login credentials",
    "received phishing email asking for password",
    "my debit card cloned at ATM",
    "hacker broke into my laptop remotely",
    "my website defaced by hackers",
]

harassment_data = [
    # physical harassment
    "my neighbor is constantly harassing me",
    "someone is following me everywhere",
    "being physically threatened by someone",
    "person is blocking my way daily",
    "someone is intimidating me physically",
    "neighbor banging my door at night",
    "person threatening me face to face",
    "goons hired to intimidate me",
    "being followed by unknown person",
    "someone is spying on me",
    "person is standing outside my house daily",
    "being monitored and followed constantly",
    "person following me on street",
    "neighbor shouting and threatening me",  # ← comma fixed

    # workplace harassment
    "boss is sexually harassing me",
    "colleague creating hostile work environment",
    "manager is mentally torturing me",
    "being discriminated at workplace",
    "senior colleague bullying me at office",
    "HR ignoring my harassment complaint",
    "being forced to work overtime without pay",
    "workplace discrimination based on gender",
    "boss threatening to fire me if I complain",

    # domestic harassment
    "husband is beating me regularly",
    "domestic violence by spouse",
    "in laws are mentally harassing me",
    "being tortured for dowry",
    "husband threatens to throw me out of house",
    "mother in law harassing me daily",
    "family member physically abusing me",
    "being forced into marriage",
    "husband not giving maintenance money",
    "constant mental harassment at home",

    # caste/religious harassment
    "being harassed due to my caste",
    "religious discrimination and harassment",
    "casteist slurs being hurled at me",
    "being denied service due to my religion",
    "upper caste people threatening me",
    "discrimination based on my community",
    "being prevented from entering temple due to caste",
    "caste based violence in my area",
    "minority community being targeted",
    "being harassed for inter caste marriage",

    # other harassment
    "receiving abusive calls continuously",
    "someone spreading false rumors about me",
    "being blackmailed by known person",
    "ex partner stalking and harassing me",
    "receiving death threats from neighbor",
    "teacher humiliating me in front of class",
    "landlord entering my room without permission",
    "being bullied in school",
    "someone pressuring me for money",
    "political worker threatening me",
    "landlord verbally abusing me daily",
    "being physically pushed by neighbor daily",
    "person waiting outside my house to threaten me",
    "my family is being intimidated by local goons",
    "someone is poisoning my dog to harass me",
    "landlord cutting water supply to harass me",
]

# ── Build DataFrame ────────────────────────────────────────────────────────────

data = {
    "text": theft_data + scam_data + consumer_data + cybercrime_data + harassment_data,
    "label": (
        ["theft"]      * len(theft_data) +
        ["scam"]       * len(scam_data) +
        ["consumer"]   * len(consumer_data) +
        ["cybercrime"] * len(cybercrime_data) +
        ["harassment"] * len(harassment_data)
    )
}

df = pd.DataFrame(data)

print(f"Total sentences  : {len(df)}")
print(f"Labels breakdown : {df['label'].value_counts().to_dict()}")

VALID_CASES = ["theft", "scam", "consumer", "cybercrime", "harassment"]

# ── Train/Test Split ───────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── Pipeline ───────────────────────────────────────────────────────────────────

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs"
    ))
])

pipeline.fit(X_train, y_train)

# ── Evaluation ─────────────────────────────────────────────────────────────────

y_pred = pipeline.predict(X_test)

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred))

print("── Confusion Matrix ──")
cm = confusion_matrix(y_test, y_pred, labels=VALID_CASES)
cm_df = pd.DataFrame(cm, index=VALID_CASES, columns=VALID_CASES)
print(cm_df)

# ── Save Model ─────────────────────────────────────────────────────────────────

os.makedirs("model", exist_ok=True)
with open("model/crime_classifier.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\n✅ Model saved to model/crime_classifier.pkl")

# ── Sample Predictions ─────────────────────────────────────────────────────────

test_inputs = [
    "my phone was stolen at market",
    "I was cheated in online fraud",
    "company not giving refund",
    "someone hacked my instagram",
    "boss is harassing me at work",
    "my car was stolen from parking",
    "fake job offer took my money",
    "restaurant gave me stale food",
    "my private photos leaked online",
    "husband is beating me daily",
]

print("\n── Sample Predictions ──")
for text in test_inputs:
    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    confidence = round(max(proba) * 100, 1)
    print(f"  '{text}'")
    print(f"   → {pred} ({confidence}% confidence)\n")