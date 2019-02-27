randomIdx = function (x_vec, percentage)
  #Function that returns percentage random indexes of the vector x_vec.
  len_v = length(x_vec)
  Idx_vec = seq(1,len_v, len = len_v)
  return (sample(Idx_vec, len_v * percentage, replace = FALSE, prob = NULL ))
