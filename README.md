# Entity Matching Model For Company Names

## Overview

The entity matching model is designed to predict the similarity between pairs of entity names, specifically company names. It's based on the DistilBERT transformer model and deployed as an API for easy integration into various applications.

## The Challenge
- Company names from different data sources don’t always match because of different word orders / spaces & special characters/ abbreviations/ typos/ changes in company type (GmbH -> AG) / prefixes and suffixes, etc.
- This is a common data science problem: some examples [here](https://medium.com/bcggamma/an-ensemble-approach-to-large-scale-fuzzy-name-matching-b3e3fa124e3c) and [here](https://towardsdatascience.com/python-tutorial-fuzzy-name-matching-algorithms-7a6f43322cc5)
- The data contains a large list of companies with both positive and negative matches.
- Initial dataset contains 7042846 labeled entity pairs. A small sample from this dataset is available in the data_sample.csv file.

## Getting Started

### Prerequisites

- Docker
- Python 3.8 (if running locally without Docker)

### Running

1. **Building the Docker Image**

   ```bash
   docker build -t entity_matcher .

2. **Running the Docker Container**

   ```bash
   docker run -p 80:80 entity_matcher

3. **(OR) Running without Docker Container**

   ```bash
   # Create a virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   
   # Install dependencies
   pip install -r requirements.txt
   
   cd app
   uvicorn main:app --reload

The API will now be accessible at http://localhost.

## Usage

You can make POST requests to the `/predict` endpoint with a pair of entity names. The response will include the prediction and the associated probability.

Example Request

```json
POST /predict
Content-Type: application/json

{
  "entity_1": "Example Company A",
  "entity_2": "Example Co. A"
}
```

Example Response

```json
{
  "prediction": 1,
  "probability": 0.999990
}
```
