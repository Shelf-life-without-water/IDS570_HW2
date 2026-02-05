# IDS 570 - Text as Data
# Homework Assignment Two: Sentiment Analysis and TF-IDF
#
# This script:
# 1) Computes raw-count sentiment (positive, negative, net) for two documents
# 2) Computes TF-IDF-weighted sentiment (positive, negative, net)
# 3) Exports a final comparison table as a CSV
#
# Required input text files (same two texts as Weeks 02–04):
#   - A07594__Circle_of_Commerce.txt
#   - B14801__Free_Trade.txt
#
# Notes:
# - Tokenization is done with a regex to keep only [a-z] words plus internal ' or -.
# - Stopwords are removed using a downloaded English stopword list.
# - The sentiment dictionary is Bing Liu's opinion lexicon (positive-words / negative-words).

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidytext)
  library(readr)
  library(stringr)
})

# ----------------------------
# 0) File paths / doc names
# ----------------------------
text_files <- tibble(
  doc_title = c("Circle_of_Commerce", "Free_Trade"),
  filepath  = c("A07594__Circle_of_Commerce.txt", "B14801__Free_Trade.txt")
)

# ----------------------------
# 1) Download helper files (if missing)
# ----------------------------
stopwords_url <- "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt"
stopwords_file <- "stopwords-en.txt"

pos_url <- "https://ptrckprry.com/course/ssd/data/positive-words.txt"
pos_file <- "positive-words.txt"

# Using a GitHub gist raw link for the negative list (same Bing Liu lexicon)
neg_url <- "https://gist.githubusercontent.com/mkulakowski2/4289441/raw/dad8b64b307cd6df8068a379079becbb3f91101a/negative-words.txt"
neg_file <- "negative-words.txt"

if (!file.exists(stopwords_file)) download.file(stopwords_url, stopwords_file, quiet = TRUE)
if (!file.exists(pos_file)) download.file(pos_url, pos_file, quiet = TRUE)
if (!file.exists(neg_file)) download.file(neg_url, neg_file, quiet = TRUE)

# ----------------------------
# 2) Read helper lists
# ----------------------------
stop_words <- read_file(stopwords_file) %>%
  str_split("\\s+") %>%
  unlist() %>%
  str_trim() %>%
  discard(~ .x == "") %>%
  tibble(word = .)

pos_words <- read_lines(pos_file) %>%
  str_trim() %>%
  discard(~ .x == "" || str_starts(.x, ";"))

neg_words <- read_lines(neg_file) %>%
  str_trim() %>%
  discard(~ .x == "" || str_starts(.x, ";"))

bing_lexicon <- bind_rows(
  tibble(word = pos_words, sentiment = "positive"),
  tibble(word = neg_words, sentiment = "negative")
) %>%
  mutate(word = str_to_lower(word))

# ----------------------------
# 3) Read + clean texts
# ----------------------------
docs <- text_files %>%
  mutate(text = map_chr(filepath, read_file)) %>%
  mutate(
    # Early modern English long-s replacement to match many tokenizers
    text = str_replace_all(text, "ſ", "s"),
    text = str_to_lower(text)
  )

# ----------------------------
# 4) Tokenize + remove stopwords
# ----------------------------
# Keep only alphabetic words with optional internal ' or - (e.g., don't, well-being)
word_pattern <- "[a-z]+(?:['-][a-z]+)*"

tidy_tokens <- docs %>%
  unnest_tokens(
    output = word,
    input  = text,
    token  = "regex",
    pattern = word_pattern,
    to_lower = FALSE
  ) %>%
  anti_join(stop_words, by = "word")

# Optional: total tokens after stopword removal (useful for debugging)
# head(tidy_tokens)

n_tokens <- tidy_tokens %>%
  count(doc_title, name = "n_tokens")

# ----------------------------
# 5) I. Raw-count sentiment
# ----------------------------
raw_summary <- tidy_tokens %>%
  inner_join(bing_lexicon, by = "word") %>%
  count(doc_title, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(
    raw_positive = positive,
    raw_negative = negative,
    raw_net = raw_positive - raw_negative
  ) %>%
  select(doc_title, raw_positive, raw_negative, raw_net)

# ----------------------------
# 6) II. TF-IDF–weighted sentiment
# ----------------------------
word_counts <- tidy_tokens %>%
  count(doc_title, word, name = "n")

tfidf_tbl <- word_counts %>%
  bind_tf_idf(term = word, document = doc_title, n = n)

tfidf_summary <- tfidf_tbl %>%
  inner_join(bing_lexicon, by = c("word" = "word")) %>%
  group_by(doc_title, sentiment) %>%
  summarise(tfidf_total = sum(tf_idf), .groups = "drop") %>%
  pivot_wider(names_from = sentiment, values_from = tfidf_total, values_fill = 0) %>%
  mutate(
    tfidf_positive = positive,
    tfidf_negative = negative,
    tfidf_net = tfidf_positive - tfidf_negative
  ) %>%
  select(doc_title, tfidf_positive, tfidf_negative, tfidf_net)

# ----------------------------
# 7) III. Final comparison table + export CSV
# ----------------------------
final_table <- raw_summary %>%
  left_join(tfidf_summary, by = "doc_title") %>%
  left_join(n_tokens, by = "doc_title") %>%
  arrange(doc_title)

print(final_table)

write_csv(final_table, "Homework_2_sentiment_comparison.csv")

# End.
