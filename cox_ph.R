#########################
#####DOVE2 model
library(DOVE)
library(data.table)
data <- fread("./data.csv")
res <- dove2(formula = Surv(event.time, event.status) ~ age + GENDER +
               as.factor(RACE) + as.factor(ETHNICITY) + as.factor(REGION) +
               vaccine(entry.time, vaccine.status, vaccine.time),
             data = data,
             plot = FALSE,
             changePts = 4*7,
             timePts = c(4, 16, 28, 40)*7)
res1 <- res$vaccine
ve_h <-res1$VE_h
ve_a <- res1$VE_a
write.csv(ve_h, "./ve_h.csv", row.names = FALSE)
write.csv(ve_a, "./ve_a.csv", row.names = FALSE)

