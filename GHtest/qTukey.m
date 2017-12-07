function x = qTukey(v,k,p);
%QTUKEY Tukey's q studentized range critical value.
%   X = QTUKEY(V,K,P) finds the Tukey's q studentized range critical value.
%(This modified macro based on the Fortran77 algorithm AS 190.2 Appl. Statist. (1983)
%gives a very good approximation for 0.8 < p < 0.995).
%
%   Syntax: function x = qtukey(v,k,p) 
%
%     Inputs:
% 	        v - sample degrees of freedom (must be the same for each sample).
% 	        k - number of samples.
% 	        p - cumulative probability value.
%
%  Created by A. Trujillo-Ortiz and R. Hernandez-Walls
%             Facultad de Ciencias Marinas
%             Universidad Autonoma de Baja California
%             Apdo. Postal 453
%             Ensenada, Baja California
%             Mexico.
%             atrujo@uabc.mx
%
%  May 18, 2003.
%
%  To cite this file, this would be an appropriate format:
%  Trujillo-Ortiz, A. and R. Hernandez-Walls. (2003). qtukey: Tukey's q studentized range 
%    critical value. A MATLAB file. [WWW document]. URL http://www.mathworks.com/
%    matlabcentral/fileexchange/loadFile.do?objectId=3469
%
%  References:
% 
%  Algorithm AS 190.2 (1983), Journal of Applied Statistics, 32(2)
%

if nargin < 3, 
    p = 0.95;
end

if nargin < 2, 
   error('Requires at least two arguments.');
end


t=norminv(.5+.5*p);
vmax=120; c=[0.89,0.237,1.214,1.21,1.414];

if v <=vmax;
    t=t+(t*t*t+t)/v/4;
    q=c(1)-c(2)*t;
    q=q-c(3)/v+c(4)*t/v;
    qc=t*(q*log(k-1)+c(5));
end
    x=qc;
    x = x/sqrt(2);
