##importing libraries
library(RCurl)
library(wordcloud)
library(tm)
library(NLP)
library(plyr)
library(stringr)
library(ggplot2)

##Reading the file
filepath <- "C:/Users/Aayushi Verma/Desktop/bs2012.txt"
text1 = readLines(filepath)
doc1 <- Corpus(VectorSource(text1))
class(doc1)

##text transformation
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
doc1 <- tm_map(doc1, toSpace, "/")
doc1 <- tm_map(doc1, toSpace, "@")
doc1 <- tm_map(doc1, toSpace, "\\|")

##text cleaning
# Convert the text to lower case
doc1 <- tm_map(doc1, content_transformer(tolower))
# Remove numbers
doc1 <- tm_map(doc1, removeNumbers)
# Remove english common stopwords
doc1 <- tm_map(doc1, removeWords, stopwords("english"))
# Remove your own stop word
# specify your stopwords as a character vector
doc1 <- tm_map(doc1, removeWords, c("and","good","the","per","cent","our","that","for","are","also","more","has","must","have","should","this","with","set","will","now","provide","provided","crore","include","madam","speaker","shell","year","years","percent","propose","from","develop","proposed")) 
# Remove punctuations
doc1 <- tm_map(doc1, removePunctuation)
# Eliminate extra white spaces
doc1 <- tm_map(doc1, stripWhitespace)
# Text stemming
#doc1 <- tm_map(doc1, stemDocument, language = "english")

##Build a term document matrix
dtm <- TermDocumentMatrix(doc1)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)

##find word associations
findAssocs(dtm, 'employment' ,0.40)
findAssocs(dtm, 'development' ,0.30)
findAssocs(dtm, 'poverty' ,0.30)     #no coorelation with this
findAssocs(dtm, 'women',0.50)
findAssocs(dtm, 'agriculture',0.30)
findAssocs(dtm, 'manufacturing',0.50)
findAssocs(dtm, 'skills',0.50)
findAssocs(dtm, 'industry',0.50)
findAssocs(dtm, 'youth',0.50)
findAssocs(dtm, 'education',0.50)
findAssocs(dtm, 'rural',0.50)


##word cloud
set.seed(1234)
dev.new()
wordcloud(words = d$word, freq = d$freq, min.freq = 1,max.words=250, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"))

##plot
v <- subset(v, v >= 25)
d <- data.frame(word = names(v),freq=v)
dev.new()
ggplot(d, aes(x = word, y = freq)) + geom_bar(stat = "identity") + xlab("Keywords") + ylab("Count") + coord_flip()

##clustering
#remove sparse terms and form term document matrix
dtm2 <- removeSparseTerms(dtm, sparse = 0.95)
m2 <- as.matrix(dtm2)
m2

m2[m2>=1] <- 1
# transform into a term-term adjacency matrix
termMatrix <- m2 %*% t(m2)

## Build a graph ###########################

library(igraph)
# build a graph from the above matrix
g <- graph.adjacency(termMatrix, weighted=T, mode = "undirected")
# remove loops
g <- simplify(g)
# set labels and degrees of vertices
V(g)$label <- V(g)$name
V(g)$degree <- degree(g)

# set seed to make the layout reproducible
set.seed(3952)
layout1 <- layout.fruchterman.reingold(g)
#dev.new()
#plot(g, layout=layout1)

V(g)$label.cex <- 2.2 * V(g)$degree / max(V(g)$degree)+ .2
V(g)$label.color <- rgb(0, 0, .2, .8)
V(g)$frame.color <- NA
egam <- (log(E(g)$weight)+.4) / max(log(E(g)$weight)+.4)
E(g)$color <- rgb(.5, .5, 0, egam)
E(g)$width <- egam
dev.new()
plot(g, layout=layout1)

##cluster terms
distMatrix <- dist(scale(m2))
fit <- hclust(distMatrix, method = "average")
dev.new()
plot(fit)
rect.hclust(fit, k=4) ##cut tree into 4 clusters

m3 <- t(m2)  ##transpose the matrix to cluster docs
set.seed(122)
k <- 4   ##no of clusters
kmeansResult <- kmeans(m3, k)
round(kmeansResult$centers, digits = 3)  ##cluster centers

for (i in 1:k) {
  cat(paste("cluster", i ,": ", sep = ""))
  s <- sort(kmeansResult$centers[i, ], decreasing = T)
  cat(names(s)[1:5], "\n")
}


##sentiment analysis
pos = scan('D:/ECO MISSION/opinion-lexicon-English/positive-words.txt',what='character', comment.char=';')
neg = scan('D:/ECO MISSION/opinion-lexicon-English/negative-words.txt',what='character', comment.char=';')
pos.words = c(pos, 'upgrade')
neg.words = c(neg, 'wtf', 'wait', 'waiting','epicfail', 'mechanical')

score.sentiment = function(sentences, pos.words, neg.words, .progress='none')
{
  require(plyr)
  require(stringr)
  # we got a vector of sentences. plyr will handle a list
  # or a vector as an "l" for us
  # we want a simple array of scores back, so we use
  # "l" + "a" + "ply" = "laply":
  scores = laply(sentences, function(sentence, pos.words, neg.words) {
    # clean up sentences with R's regex-driven global substitute, gsub():
    sentence = gsub('[[:punct:]]', '', sentence)
    sentence = gsub('[[:cntrl:]]', '', sentence)
    sentence = gsub('\\d+', '', sentence)
    # and convert to lower case:
    sentence = tolower(sentence)
    # split into words. str_split is in the stringr package
    word.list = str_split(sentence, '\\s+')
    # sometimes a list() is one level of hierarchy too much
    words = unlist(word.list)
    # compare our words to the dictionaries of positive & negative terms
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    # match() returns the position of the matched term or NA
    # we just want a TRUE/FALSE:
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
    score = sum(pos.matches) - sum(neg.matches)
    return(score)
  }, pos.words, neg.words, .progress=.progress )
  scores.df = data.frame(score=scores, text=sentences)
  return(scores.df)
}
budget.scores = score.sentiment(text1, pos.words,neg.words, .progress='text')
budget.scores
sentiment_score_2014 <- table(budget.scores$score)
hist(sentiment_score_2014) ##histogram
dev.new()
plot(sentiment_score_2014) ##plot





