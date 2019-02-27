#Script equivalent to clear workspace and window like clear all, clc in MATLAB
rm(list=ls())
graphics.off()
cat("\014")