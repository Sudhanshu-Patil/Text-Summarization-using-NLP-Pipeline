## Link to Colab repo: https://colab.research.google.com/drive/1-vOnrSg2QHYWvZi6z3MJApHURUVjOIgt?usp=sharing

## Build the Docker Image

```sh
docker build -t ner-extraction-service .
```
## Run the Docker container

```sh
docker run -p 5000:5000 ner-extraction-service

```
## Sample CURL Command

```
curl -X POST http://localhost:5000/extract -H "Content-Type: application/json" -d '{"snippet": "We love the analytics, but CompetitorX has a cheaper subscription."}'
```
