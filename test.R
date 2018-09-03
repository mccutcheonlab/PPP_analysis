message(getwd())
for(i in 1:100) {
  cat(".")
  Sys.sleep(0.01)
}
message("nBye.")
Sys.sleep(3)