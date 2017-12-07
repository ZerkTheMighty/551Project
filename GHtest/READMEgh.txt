GHTEST Games-Howell's approximate test of equality of means from normal populations when 
variances are heterogeneous. It is a related test associated to the Behrens-Fisher problem.
It uses the Tukey's studentized range with specially weighted average degrees of freedom (df')
and a standard error based on the averages of the variances of the means.
(Unplanned comparisions among pairs of means using the Games and Howell method. For the
statistical test the function calls the file qTukey.m)

-------------------------------------------------------------
Created by A. Trujillo-Ortiz and R. Hernandez-Walls
           Facultad de Ciencias Marinas
           Universidad Autonoma de Baja California
           Apdo. Postal 453
           Ensenada, Baja California
           Mexico.
           atrujo@uabc.mx

June 30, 2003.

To cite this file, this would be an appropriate format:
Trujillo-Ortiz, A. and R. Hernandez-Walls. (2003). GHtest: Games-Howell's approximate
test of equality of means from normal population when variances are heterogeneous. 
A MATLAB file. [WWW document]. URL http://www.mathworks.com/matlabcentral/fileexchange/
loadFile.do?objectId=3676&objectType=FILE
-------------------------------------------------------------
Congratulations on deciding to use this MATLAB macro file.  
This program has been developed to help you quickly calculate the
approximate test of equality of means from normal population when 
variances are heterogeneous.
-------------------------------------------------------------
This zip file is free; you can redistribute it and/or modify at your option.
-------------------------------------------------------------
This zip file contains....
	List of files you should need

GHtest.m      Games-Howell's approximate test of equality of means when variances are heterogeneous.
qTukey.m      Tukey's q studentized range critical value.
READMEgh.TXT		
-------------------------------------------------------------
Usage

1. It is necessary you have defined on Matlab the X - data matrix [1=yes (default); 2=no (If yes,
size of matrix (X) must be n-by-2; data=column 1, sample=column 2. If not, you must to give the 
statistics matrix (X) k-by-3 (sample=column 1, sample sizes=column 2, means=column 3,
variances=column 4))]. 

2. For running this file it is necessary to call the GHtest function as GHtest(X,d,alpha)
(d-data matrix [1=yes (default); 2=no]; alpha-significance level default = 0.05). Please see the 
help GHtest.

3. Once you input your choices, it will appears your results.
-------------------------------------------------------------
We claim no responsibility for the results that are obtained 
from your data using this file.
-------------------------------------------------------------
Copyright.2003