MSE = function (y_pred,y_vec)
{
	train_err = sqrt(sum((y_vec - y_pred)^2)/length(y_vec))
	return(train_err)
}