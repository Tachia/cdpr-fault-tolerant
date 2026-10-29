# Container with the environment of the stored results (Python 3.13, the libraries of
# requirements.txt, gcc for the compiled solvers). Not tested on the machine where the results
# were produced, which had no access to a container registry; see REPRODUCE.md.
#   docker build -t cdpr_ft .
#   docker run --rm -it cdpr_ft python reproduce.py quick
#   docker run --rm -it -v "$PWD/reproduction:/cdpr_ft/reproduction" cdpr_ft python reproduce.py phase3
FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /cdpr_ft
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "reproduce.py", "check"]
