library(SamplingStrata)

id <- seq(1:length(z_h))
domain <- rep(1, length(z_h))
d <- data.frame(id = id, y1 = z_p1, y2 = z_p2, x=z_h, domain=domain)

frame1 <- buildFrameDF(df = d,
                       id = "id",
                       X = c("x"),
                       Y = c("y1", "y2"),
                       domainvalue = "domain" )

strata1 <- buildStrataDF(frame1, progress=F)

cv <- as.data.frame(list(DOM=rep("DOM1",1),
                         CV1=rep(0.10,1),
                         CV2=rep(0.10,1),
                         domainvalue=c(1:1)))

checkInput(errors = checkInput(errors = cv, 
                               strata = strata1, 
                               sampframe = frame1))

allocation <- bethel(strata1,cv[1,])
sum(allocation)

d$progr <- c(1:nrow(d))
frame2 <- buildFrameDF(df = d,
                       id = "id",
                       X = "progr",
                       Y = c("y1", "y2"),
                       domainvalue = "domain")

frame3 <- buildFrameDF(df = d,
                       id = "id",
                       X = c("x"),
                       Y = c("y1", "y2"),
                       domainvalue = "domain")

set.seed(1234)
init_sol3 <- KmeansSolution2(frame=frame3,
                             errors=cv,
                             maxclusters = 10)

nstrata3 <- tapply(init_sol3$suggestions,
                   init_sol3$domainvalue,
                   FUN=function(x) length(unique(x)))

initial_solution3 <- prepareSuggestion(init_sol3,frame3,nstrata3)

set.seed(1234)
solution3 <- optimStrata(method = "continuous",
                         errors = cv, 
                         framesamp = frame3,
                         iter = 50,
                         pops = 10,
                         nStrata = nstrata3,
                         suggestions = initial_solution3);

strataStructure <- summaryStrata(solution3$framenew,
                                 solution3$aggr_strata,
                                 progress=FALSE)

wl_OBS <- data.frame(h_y(c(res$OSB)))