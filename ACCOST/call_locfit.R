
library(locfit)

do_fit <- function(x, y){
    #print("X for fit")
    #print(x)
    #print("Y for fit")
    #print(y)
    fit_locfit <- locfit.raw(x, y, family="gamma", maxk=200)
    print(summary(fit_locfit))
    return(fit_locfit)
}

safepredict <- function(fit, x) {
    result <- rep.int(NA_real_, length(x))
    x = as.numeric(x)
    #print("X for predict:")
    #print(x[is.finite(x)])
    #print("fit")
    print(summary(fit))
    result[is.finite(x)] <- predict(fit, x[is.finite(x)], se.fit=F)
    #print(result)
    return(result)
}

