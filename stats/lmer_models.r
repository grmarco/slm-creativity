setwd('/home/gmremon/onedrive-uned/workspace/synopsis-english-spanish/english/english-human-assesment/lmer')
df <- read.csv('answers.csv')

df$question <- as.factor(df$question)
df$user <- as.factor(df$user)
df$writer <- as.factor(df$writer)
df$gender <- as.factor(df$gender)
df$education <- as.factor(df$education)
df$language <- as.factor(df$language)

dfnorm <- subset(df,df['experiment']=='main')

library(lmerTest)

mixed0 <- lmer(answer ~ writer + (1|title) + (1|user), data = dfnorm)
mixed1 <- lmer(answer ~ writer + language + (1|title) + (1|user), data = dfnorm)
mixed2 <- lmer(answer ~ writer + education + (1|title) + (1|user), data = dfnorm)
mixed3 <- lmer(answer ~ writer + age + (1|title) + (1|user), data = dfnorm)
mixed4 <- lmer(answer ~ writer + question + (1|title) + (1|user), data = dfnorm)
anova(mixed0,mixed1)
anova(mixed0,mixed2)
anova(mixed0,mixed3)
anova(mixed0,mixed4)

mixed5 <- lmer(answer ~ writer + question + (1|title) + (writer|user), data = dfnorm)
anova(mixed4,mixed5)

mixed6 <- lmer(answer ~ writer*question + (1|title) + (writer|user), data = dfnorm)
anova(mixed5,mixed6)

mixed7 <- lmer(answer ~ (writer+language+education)*question + (1|title) + (writer|user), data = dfnorm)
anova(mixed6,mixed7)

summary(mixed6)
summary(mixed7)

mixed_final <- lmer(answer ~ writer*question + experiment + (1|title) + (writer|user), data = df)
summary(mixed_final)
