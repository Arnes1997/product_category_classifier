# Product Category Classifier

Ovaj projekat predstavlja **model za automatsku klasifikaciju proizvoda u kategorije** na osnovu naziva proizvoda i nekoliko jednostavnih numeričkih značajki. Projekat koristi **Python**, **pandas**, **scikit-learn** i **joblib** za treniranje, spremanje i testiranje modela.

## 📝 Dataset

- Dataset sadrži više od 30.000 proizvoda.
- Svaki proizvod ima sledeće kolone:
  - `Product ID`
  - `Product Title`
  - `Merchant ID`
  - `Category Label` (ciljna kategorija)
  - `Product Code`
  - `Number of Views`
  - `Merchant Rating`
  - `Listing Date`

---

## ⚙️ Koraci koje sam poduzeo

1. **Učitavanje i čišćenje podataka**
   - Uklonjene prazne vrijednosti u koloni `Category Label`.

2. **Kreiranje numeričkih značajki**
   - `title_length` – dužina naslova proizvoda (broj karaktera).
   - `num_words` – broj riječi u naslovu.
   - `has_digits` – da li naslov sadrži cifre.
   - `has_uppercase` – da li naslov sadrži velika slova.

3. **Priprema TF-IDF značajki**
   - TF-IDF vektorizacija naziva proizvoda.
   - Ograničenje na 5000 najvažnijih n-grama (1,2).

4. **Kombinacija značajki u pipeline**
   - Numeričke značajke standardizovane (`StandardScaler`).
   - Tekstualne značajke transformisane TF-IDF-om.
   - Sve kombinovano u `ColumnTransformer`.

5. **Treniranje modela**
   - Testirano više modela:
     - Logistic Regression
     - Random Forest Classifier
     - Support Vector Classifier (SVC)
   - Random Forest dao najbolju ukupnu tačnost (~94–95%).

6. **Spremanje modela**
   - Trenirani pipeline sačuvan koristeći `joblib` kao `product_classifier.pkl`.

7. **Testiranje modela**
   - Napravljena skripta `model_testing.py` koja:
     - Testira model na unaprijed definisanim proizvodima.
     - Omogućava **interaktivno testiranje**: korisnik može unositi proizvode i dobiti predviđene kategorije.



