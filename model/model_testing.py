import pandas as pd
import joblib
import re

# Učitavanje modela
model = joblib.load('product_classifier.pkl')

# Funkcija za kreiranje numeričkih značajki iz naslova
def extract_features(title):
    return pd.DataFrame({
        'Product Title': [title],
        'title_length': [len(title)],
        'num_words': [len(title.split())],
        'has_digits': [int(bool(re.search(r'\d', title)))],
        'has_uppercase': [int(any(w.isupper() for w in title.split()))]
    })

# Testiranje na 5 primjera
product_titles = [
    'Samsung Galaxy S21 128GB',
    'Bosch Dishwasher 12 sets',
    'Apple MacBook Air M2 256GB',
    'LG Fridge Freezer 300L',
    'Sony 55-inch OLED TV'
]

test_data = pd.concat([extract_features(title) for title in product_titles], ignore_index=True)
predictions = model.predict(test_data)

print("=== Predikcije za 5 test proizvoda ===")
for title, category in zip(product_titles, predictions):
    print(f"Naziv proizvoda: {title} -- Predikcija: {category}")

# Interaktivno testiranje
print("\n=== Interaktivno testiranje modela ===")
while True:
    user_input = input("Unesi naziv proizvoda (ili 'exit' za izlaz): ")
    if user_input.lower() == 'exit':
        print("Izlaz iz programa.")
        break
    user_test = extract_features(user_input)
    pred_category = model.predict(user_test)[0]
    print(f"Predviđena kategorija: {pred_category}\n")
