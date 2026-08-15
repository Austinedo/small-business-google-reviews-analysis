# Topic Modeling and Analysis for Small Business Reviews

![Python](https://img.shields.io/badge/Python-3.10.19-blue)
![Tags](https://img.shields.io/badge/Tags-NLP%20%7C%20Transformers%20%7C%20Topic%20Modeling%20%7C%20Visualization%20%7C%20Machine%20Learning-red)

An end-to-end natural language processing (NLP) project that analyzes negative Google Reviews for an anonymized business in the beauty and hair-care service industry. The project compares lexical clustering, semantic clustering, and probabilistic topic modeling to uncover recurring operational issues hidden in customer feedback.

## Project Overview

Online reviews contain detailed feedback about service quality, staff interactions, appointment experiences, wait times, product outcomes and more. Star ratings alone summarize sentiment, but they do not identify the operational issues responsible for driving customer's experience.

This project focuses on reviews with ratings of **3 stars or lower** and applies unsupervised learning to answer the following question:

> What recurring themes drive negative customer experiences, and which modeling approach provides the most useful and interpretable summary of those themes?

The analysis implements and evaluates three approaches:

1. **K-Means clustering with TF-IDF features** — groups reviews using lexical similarity and word-frequency patterns.
2. **K-Means clustering with SBERT embeddings** — groups reviews according to semantic similarity, including conceptually similar language with different word choices.
3. **Latent Dirichlet Allocation (LDA)** — models each review as a mixture of latent topics, allowing a single review to discuss multiple concerns.

## Data

- **Source:** Google Reviews collected with a [Google Reviews scraping tool](https://github.com/georgekhananaev/google-reviews-scraper-pro) developed by George Khanaev.
- **Domain:** An anonymized local business in the beauty and hair-care service industry.
- **Unit of analysis:** Written customer reviews.
- **Target subset:** Negative reviews, defined as ratings less than or equal to 3 out of 5. (Subset size was 462 reviews)

To protect privacy, business identifiers and personally identifiable reviewer information were removed before analysis. Retained fields were limited to information relevant to the text and rating analysis.

## Workflow

```text
                   Google Reviews
                         |
                         v
               General text cleaning
    (HTML/web artifacts, URLs, email addresses)
                         |
                         v
         Filter reviews with ratings <= 3     
                         |
     +-------------------+-------------------+
     |                   |                   |
     v                   v                   v
TF-IDF + K-Means    SBERT + K-Means        LDA
Lexical clusters    Semantic clusters      Mixed-membership topics
```

### Preprocessing

A common initial cleaning stage removed HTML/web artifacts, URLs, and email addresses. The resulting negative-review corpus was then prepared differently for each modeling approach:

| Modeling branch | Representation and preprocessing |
|-----|---|
| TF-IDF + K-Means | Lowercasing, removal of punctuation, digits, artifacts, and stopwords; transformed into a sparse TF-IDF document-term matrix |
| SBERT + K-Means | General-cleaned review text was encoded directly with a pretrained Sentence-BERT model to preserve wording and contextual meaning |
| LDA | Tokenization, lemmatization, standard and custom stopword filtering, n-gram extraction, and construction/tuning of a bag-of-words dictionary |

## Methods

### 1. TF-IDF + K-Means

TF-IDF represents each review by the importance of its words relative to the review and the full corpus. K-Means then partitions reviews into hard clusters based on distance in that high-dimensional feature space.

The selected TF-IDF configuration used:

```python
max_features = 100
min_df = 11
max_df = 0.6
n_clusters = 3
random_state = 6740
```

This baseline was useful for testing whether prominent word patterns could reveal topics. Model selection considered the within-cluster sum of squares (WCSS), elbow plots, silhouette scores, cluster keywords, and representative reviews.

### 2. SBERT + K-Means

Sentence-BERT (SBERT) creates dense sentence embeddings that capture contextual and semantic similarity beyond exact word overlap. Embeddings were L2-normalized before K-Means clustering, making squared Euclidean distance monotonic with cosine similarity:

$$
\|\|x-y\|\|_2^2 = 2 - 2\cos(\theta)
$$

for unit-normalized vectors \(x\) and \(y\). This makes K-Means suitable for grouping reviews by cosine-based semantic similarity.

The selected solution used **2 clusters** and was evaluated with elbow and silhouette diagnostics, cluster sizes, similarity to cluster centroids, and representative reviews.

### 3. Latent Dirichlet Allocation

LDA is a probabilistic topic model in which:

- Each review can contain a mixture of topics.
- Each topic is represented as a distribution over words.

Unlike K-Means, which assigns every review to one cluster, LDA accommodates the fact that a review may discuss several concerns at once—for example, an unsatisfactory haircut, poor communication, and a long wait.

Vocabulary filtering and preprocessing were iteratively tuned. Topic-number selection used coherence analysis, and a **3-topic LDA model** was selected because it provided the strongest balance of quantitative coherence and human interpretability.

## Results

### TF-IDF + K-Means: weak lexical separation

The TF-IDF K-Means analysis did not produce clear, interpretable topical clusters.

- The WCSS curve did not show a distinct elbow.
- Silhouette scores were consistently low across tested values of \(k\) from 2 to 9.
- Cluster keyword lists overlapped substantially and contained repeated high-frequency terms such as *hair*, *haircut*, *service*, *place*, and *staff*.
- The result suggested that sparse lexical features and single-cluster assignments did not adequately capture the overlapping structure of customer complaints.

### SBERT + K-Means: more meaningful semantic groupings

SBERT embeddings improved interpretability relative to TF-IDF by grouping conceptually similar reviews even when they used different language.

The selected \(k=2\) solution identified two broad, actionable groupings:

| Cluster | Interpretation |
|----|---|
| Cluster 0 | Unprofessional, rude, or inappropriate staff behavior and communication |
| Cluster 1 | Service-quality concerns, including disappointing haircut outcomes, rushed appointments, and dissatisfaction with the overall service experience |

Although silhouette diagnostics still indicated weak global cluster separation—particularly at larger values of \(k\)—the small-cluster solution produced semantically meaningful groupings. This indicates that contextual embeddings offer a stronger representation than lexical TF-IDF features for this review corpus.

### LDA: clearest and most coherent topic structure

The three-topic LDA model produced the most interpretable thematic summary of negative reviews. The final topics were:

| Topic | Main theme | Operational interpretation |
|----|---|---|
| Topic 1 | Service mistakes and unsatisfactory haircut outcomes | Uneven or poor-quality haircuts, frustration, and customers deciding not to return or recommend the business |
| Topic 2 | Wait times and the in-salon experience | Delays, being unattended to, poor customer service, and feeling uncared for during the visit |
| Topic 3 | Unprofessional behavior and communication | Rude, disrespectful, uncomfortable, or inappropriate interactions with barbers or staff |

The LDA topics showed strong coherence and clear semantic separation after dictionary and preprocessing tuning. These themes also aligned with the SBERT cluster interpretations, increasing confidence that they reflect stable patterns in the review corpus rather than artifacts of a single method.

## Key Findings

- Negative reviews are dominated by a small set of operational concerns: **service quality**, **staff behavior**, and the **appointment/waiting experience**.
- Hard clustering with K-Means struggled because customer reviews commonly contain multiple themes and do not naturally partition into sharply distinct groups.
- SBERT-based clustering was more interpretable than TF-IDF-based clustering because it captured semantic similarity rather than relying only on shared words.
- LDA provided the best final approach because mixed-membership topic modeling matches the structure of customer reviews: one review can describe multiple operational failures.

## Business Implications

The results suggest three practical improvement areas for a service business:

1. **Improve service consistency and quality control** by addressing haircut/service errors, confirming customer expectations before service, and using post-service quality checks.
2. **Strengthen staff communication and professionalism** through customer-service training, respectful communication standards, and clear escalation procedures for negative interactions.
3. **Improve appointment flow and wait-time management** by setting clearer expectations, improving queue management, and ensuring customers feel acknowledged while waiting.

## Repository Structure

The repository is intended to contain materials such as:

```text
|
├── data/                    # Processed/anonymized review data (if distribution is permitted)
├── figures/                 # Figures, topic visualizations, and model artifacts
├── notebooks/               # Data cleaning, EDA, clustering, and LDA analysis
├── reports/                 # Formal Document with further explanations + deep-dives
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

> Raw review data and identifying information is not published to this repository

## Installation

Create and activate a virtual environment, then install the repository dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the required NLP resources if they are not already installed:

```python
import nltk
nltk.download("stopwords")
```

If the notebook uses a spaCy English pipeline, install it separately:

```bash
python -m spacy download en_core_web_sm
```

## Technologies

- Python
- Pandas and NumPy
- scikit-learn
- NLTK and spaCy
- Sentence-Transformers / PyTorch
- Gensim
- pyLDAvis
- Plotly and Matplotlib
- Yellowbrick

## Limitations and Future Work

This project analyzes an anonymized business and focuses only on reviews rated 3 stars or lower. The results should therefore be interpreted as themes within negative feedback rather than as a complete picture of customer sentiment.

Potential extensions include:

- Compare positive and negative reviews to identify drivers of satisfaction and dissatisfaction.
- Add domain-specific vocabulary and preprocessing rules for the beauty and hair-care industry.
- Evaluate alternative topic-modeling methods, such as BERTopic or neural topic models.
- Track topic prevalence over time to identify emerging operational issues.
- Combine review topics with ratings, responses, and engagement information to prioritize improvements.

## Author

**Austine Do**

Completed July 31, 2026.
