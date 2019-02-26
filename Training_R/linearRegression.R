linearRegression = function (x_vec,y_vec,w_vec)
{
  # matrix assembly
  m11 = sum(w_vec)
  m21 = sum(w_vec*x_vec)
  m22 = sum(w_vec*(x_vec^2))
  #
  M = matrix(c(m11,m21,m21,m22),nrow=2,ncol=2,byrow=T)  
  # rhs
  b = c(sum(w_vec*y_vec),sum(w_vec*x_vec*y_vec))
  # theta
  theta_vec = solve(M,b)

  return(theta_vec) 
}