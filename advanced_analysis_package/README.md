#advanced_analysis_package

This is a python package to perform analysis , outlier treatment and variable reduction on pandas dataframe. List of functions include:-

analyze - a. edd - replica of sas edd maco with new features of coelation and p values when dependent variable is given.return a pandas dataframe. b. graphical_analysis - gives bivariate ananlysis of variable with respect to dependent variable. Display and saves plots and tables at equied path. c. numerical_categorical_division - return list of numeric and categoical variables in pandas dataframe
variable_treatment a. exponential_smoothning - outlier treatment for variable b.capping_and_flooring - outlier teatment for variable c.make_dummies - create dummies for categorical variable(one hot encoding) d.make_dummies_binary - create binary dummies for variable
variable_reduction a. inter_correlation_clusters - return clusters of variables based on cutoff correlation b. varclus - return list of columns 1 from each cluster c. vif_reduction - return list of columns to be dropped by vif_eduction method d. backward_selection - returns list of columns to be dropped by backwad selection