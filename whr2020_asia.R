# WHR 2020 Clustering

# load libraries
library(dplyr)
library(psych)
library(ggplot2)
library(ggpubr)
library(gridExtra)
library(cluster)
library(factoextra)
library(reshape2)

# load data
wh <- read.csv(file.choose())
wh <- wh %>% select(1:3,7:12) 
wh <- wh %>% mutate(GDP.per.capita=exp(Logged.GDP.per.capita))
wh <- wh[-4]
wh_asia <- wh %>% filter(grepl('Asia', Regional.indicator))
# print countries and region
wh_asia %>% select(Country.name, Regional.indicator) %>% print.data.frame()

# Exploratory data analysis
# correlation matrix
pairs.panels(wh_asia[,-c(1,2)], scale=TRUE)

# country bar plots
grid.arrange(
  ggplot(wh_asia, aes(reorder(x=factor(Country.name), Ladder.score, FUN=min), y=Ladder.score, fill=Regional.indicator)) + geom_col() + coord_flip() + labs(x="", y="") + scale_fill_manual(values=c("#00AFBB", "#E7B800", "#FC4E07")) + theme_classic()+ ggtitle("Ladder score")+ theme(legend.position = "none"),
  
  ggplot(wh_asia, aes(reorder(x=factor(Country.name), Social.support, FUN=min), y=Social.support, fill=Regional.indicator)) + geom_col() + coord_flip() + labs(x="",y="") + scale_fill_manual(values=c("#00AFBB", "#E7B800", "#FC4E07")) + theme_classic() + theme(legend.position = "none")+ ggtitle("Social support"),
  
  ggplot(wh_asia, aes(reorder(x=factor(Country.name), Healthy.life.expectancy, FUN=min), y=Healthy.life.expectancy, fill=Regional.indicator)) + geom_col() + coord_flip() + labs(x="",y="") + scale_fill_manual(values=c("#00AFBB", "#E7B800", "#FC4E07")) + theme_classic() + theme(legend.position = "none")+ ggtitle("Healthy life expectancy"),
  
  ggplot(wh_asia, aes(reorder(x=factor(Country.name), Freedom.to.make.life.choices, FUN=min), y=Freedom.to.make.life.choices, fill=Regional.indicator)) + geom_col() + coord_flip() + labs(x="", y="") + scale_fill_manual(values=c("#00AFBB", "#E7B800", "#FC4E07")) + theme_classic() + theme(legend.position = "none") +
    ggtitle("Freedom to make life choices"),
  
  ggplot(wh_asia, aes(reorder(x=factor(Country.name), Generosity, FUN=min), y=Generosity, fill=Regional.indicator)) + geom_col() + coord_flip() + labs(x="",y="") + scale_fill_manual(values=c("#00AFBB", "#E7B800", "#FC4E07")) + theme_classic() + theme(legend.position = "none")+  ggtitle("Generosity"),
  
  ggplot(wh_asia, aes(reorder(x=factor(Country.name), Perceptions.of.corruption, FUN=min), y=Perceptions.of.corruption, fill=Regional.indicator)) + geom_col() + coord_flip() + labs(x="", y="") + scale_fill_manual(values=c("#00AFBB", "#E7B800", "#FC4E07")) + theme_classic() + theme(legend.position = "none")+ ggtitle("Perceptions of corruption"),
  
  ggplot(wh_asia, aes(reorder(x=factor(Country.name), GDP.per.capita, FUN=min), y=GDP.per.capita, fill=Regional.indicator)) + geom_col() + coord_flip() + labs(fill="Region",x="", y="") + scale_fill_manual(values=c("#00AFBB", "#E7B800", "#FC4E07")) + theme_classic() + theme(legend.position = "none")+  ggtitle("GDP per capita")+ theme(legend.position = "right"),
  
  nrow=3
)

# scatter plot with boxplots by region
grid.arrange(
  
  ggscatterhist(
    wh_asia, x = "Social.support", y = "Ladder.score",
    color = "Regional.indicator", size = 3, alpha = 0.6,
    palette = c("#00AFBB", "#E7B800", "#FC4E07"),
    margin.plot = "boxplot",
    margin.params = list(fill = "Regional.indicator", color = "black", size = 0.2),
    ggtheme = theme_bw(),
    legend = "none",
    title = "Ladder score vs. Social support"
  ),
  
  ggscatterhist(
    wh_asia, x = "GDP.per.capita", y = "Healthy.life.expectancy",
    color = "Regional.indicator", size = 3, alpha = 0.6,
    palette = c("#00AFBB", "#E7B800", "#FC4E07"),
    margin.plot = "boxplot",
    margin.params = list(fill = "Regional.indicator", color = "black", size = 0.2),
    ggtheme = theme_bw(),
    title = "Healthy life expectancy vs. GDP per capita"
  ),
  
  ggscatterhist(
    wh_asia, x = "GDP.per.capita", y = "Perceptions.of.corruption",
    color = "Regional.indicator", size = 3, alpha = 0.6,
    palette = c("#00AFBB", "#E7B800", "#FC4E07"),
    margin.plot = "boxplot",
    margin.params = list(fill = "Regional.indicator", color = "black", size = 0.2),
    ggtheme = theme_bw(),
    legend = "none",
    title = "Perceptions of corruption vs. GDP per capita"
  ),
  ncol=3
)

# prepare data for clustering
# standardize data
z <- wh_asia[,-c(1,2)] 
means <- apply(z,2,mean) # calculate mean for columns
sds <- apply(z,2,sd) # calculate standard deviation for columns
nor <- scale(z,center=means,scale=sds) # standardize

# calculate distance matrix
distance = dist(nor) # calculate the eulcledian distances
round(distance,2) 

# random number generator
set.seed(8)

# WSS plot
# calculate within group sum of squares
wss <- (nrow(nor)-1)*sum(apply(nor,2,var)) 
for (i in 2:20) wss[i] <- sum(kmeans(nor, centers=i)$withinss)
plot(1:20, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares", main="WSS Plot") 

# K-means clustering
kc<-kmeans(nor,4)
# cluster memberships
wh_asia$Cluster <- kc$cluster
cat("Cluster 1:",paste0(subset(wh_asia$Country.name, wh_asia$Cluster==1),", ", collapse="") %>% trimws(whitespace = ", "))
cat("Cluster 2:",paste0(subset(wh_asia$Country.name, wh_asia$Cluster==2),", ", collapse="") %>% trimws(whitespace = ", "))
cat("Cluster 3:",paste0(subset(wh_asia$Country.name, wh_asia$Cluster==3),", ", collapse="") %>% trimws(whitespace = ", "))
cat("Cluster 4:",paste0(subset(wh_asia$Country.name, wh_asia$Cluster==4),", ", collapse="") %>% trimws(whitespace = ", "))
# k-means cluster plot
fviz_cluster(list(data = nor, cluster = kc$cluster), ggtheme = theme_classic()) + ggtitle("K-Means Cluster Plot")
kc
# characterizing clusters
# dot plot of standardized averages
kc.means.long <- melt(t(aggregate(nor, list(kc$cluster), mean)))[-c(1,9,17,25),] %>%
  rename(category=Var1, cluster=Var2)
ggplot(kc.means.long, aes(x = factor(category), y = value, group = as.factor(cluster), color = as.factor(cluster))) +
  geom_point(size=3)  + labs(color='Cluster', x="", y="") + ggtitle("Standardized Averages") +
  coord_flip()
# silhouette plot
plot(silhouette((kc$cluster), distance), 
     col=c("#F8766D","#7CAE00","#00BFC4","#C77CFF"), 
     main = "Silhouette plot")

# Hierarchical agglomerative clustering with complete linkage
wh_asia.hclust.c <- hclust(distance, method = "complete")
# hiearchial cluster membership
member.c <- cutree(wh_asia.hclust.c,4)
# dendrogram and cluster plot
plot(wh_asia.hclust.c,labels=wh_asia$Country.name,main='Complete Linkage k=4', hang=-1)
rect.hclust(wh_asia.hclust.c, k=4, border= c("#C77CFF","#7CAE00","#00BFC4","#F8766D"))
fviz_cluster(list(data = nor, cluster = member.c), ggtheme = theme_classic()) + ggtitle("Complete Linkage Cluster Plot")
# silhouette plot
plot(silhouette(cutree(wh_asia.hclust.c, 4), distance), 
     col=c("#F8766D","#7CAE00","#00BFC4","#C77CFF"), 
     main = "Complete Linkage Silhouette Plot")

# Hierarchical agglomerative clustering with average linkage 
wh_asia.hclust.a <- hclust(distance,method="average")
# hierarchical cluster memberships
member.a <- cutree(wh_asia.hclust.a,4)
# dendrogram and cluster plot
plot(wh_asia.hclust.a,labels=wh_asia$Country.name,main='Average Linkage k=4', hang=-1)
rect.hclust(wh_asia.hclust.a, k=4, border= c("#C77CFF","#7CAE00","#F8766D","#00BFC4"))
fviz_cluster(list(data = nor, cluster = member.a), ggtheme = theme_classic()) + ggtitle("Average Linkage Cluster Plot")
# silhouette plot
plot(silhouette(cutree(wh_asia.hclust.a, 4), distance), 
     col=c("#F8766D","#7CAE00","#00BFC4","#C77CFF"), 
     main = "Average Linkage Silhouette plot")

