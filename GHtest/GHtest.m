function [GHtest] = GHtest(X,d,alpha)
%GHTEST Games-Howell's approximate test of equality of means from normal population when
%variances are heterogeneous. It is a related test associated to the Behrens-Fisher problem.
%It uses the Tukey's studentized range with specially weighted average degrees of freedom (df')
%and a standard error based on the averages of the variances of the means.
%(Unplanned comparisions among pairs of means using the Games and Howell method. For the
%statistical test the function calls the file qTukey.m)
%
%   Syntax: function [GHtest] = GHtest(X,d,alpha) 
%      
%     Inputs:
%          d - data matrix [1=yes (default); 2=no (If yes, size of matrix (X) must be n-by-2; data=column 1, sample=column 2.
%              If not, you must to give the statistics matrix (X) k-by-3 (sample=column 1, sample sizes=column 2, means=column 3,
%              variances=column 4))]. 
%      alpha - significance level (default = 0.05).
%     Outputs:
%          - Complete table of the statistical analysis of the difference between each pair of means.
%
%    Example: From the example on Box 13.2 of Sokal and Rohlf (1981, pp.408-410), to test the
%             homogeneity among the means when variances are heterogeneous with a significance
%             level = 0.05.
%
%                               Statistic
%                   -------------------------------------
%                   Sample    n       Mean       Variance
%                   -------------------------------------
%                      1     18       3.88        0.0707
%                      2     13       4.61        0.1447
%                      3     17       4.79        0.0237
%                      4     16       4.73        0.0836
%                      5      8       4.92        0.2187
%                      6     11       4.96        0.1770
%                      7     10       5.20        0.0791
%                      8     10       6.58        0.2331
%                   -------------------------------------
%                                       
%           Data matrix must be:
%            X=[1 18 3.88 0.0707;2 13 4.61 0.1447;3 17 4.79 0.0237;4 16 4.73 0.0836;
%            5 8 4.92 0.2189;6 11 4.96 0.1770;7 10 5.20 0.0791;8 10 6.58 0.2331];
%
%     Calling on Matlab the function: 
%             GHtest(X,2)
%
%       Answer is:
%
% You have choosed the statistics option.
% 
% The number of samples are: 8
%
% Table of the difference between each pair of means in terms of the Tukey's
% statistic and then comparing them with its critical value.
% ------------------------------------------------------------------------
% Comparison            D               q              df'     Conclusion
% ------------------------------------------------------------------------
%    8  7             1.3800          7.8102           14          S  
%    8  6             1.6200          8.1613           17          S  
%    8  5             1.6600          7.3743           15          S  
%    8  4             1.8500         10.9517           13          S  
%    8  3             1.7900         11.3885           10          S  
%    8  2             1.9700         10.6152           16          S  
%    8  1             2.7000         16.3598           12          S  
%    7  6             0.2400          1.5492           17          NS 
%    7  5             0.2800          1.4909           10          NS 
%    7  4             0.4700          4.1009           19          NS 
%    7  3             0.4100          4.2506           12          NS 
%    7  2             0.5900          4.2757           20          NS 
%    7  1             1.3200         12.1322           17          S  
%    6  5             0.0400          0.1919           14          NS 
%    6  4             0.2300          1.5753           16          NS 
%    6  3             0.1700          1.2856           11          NS 
%    6  2             0.3500          2.1213           20          NS 
%    6  1             1.0800          7.6332           14          S  
%    5  4             0.1900          1.0525            9          NS 
%    5  3             0.1300          0.7666            7          NS 
%    5  2             0.3100          1.5800           12          NS 
%    5  1             1.0400          5.8793            9          S  
%    4  3             0.0600          0.7375           22          NS 
%    4  2             0.1200          0.9383           22          NS 
%    4  1             0.8500          8.8847           30          S  
%    3  2             0.1800          1.6084           15          NS 
%    3  1             0.9100         12.4741           27          S  
%    2  1             0.7300          5.9488           20          S  
% ------------------------------------------------------------------------
% With a given significance level of: 0.0500
% The results are significant (S) and/or not significant (NS).
%

%  Created by A. Trujillo-Ortiz and R. Hernandez-Walls
%             Facultad de Ciencias Marinas
%             Universidad Autonoma de Baja California
%             Apdo. Postal 453
%             Ensenada, Baja California
%             Mexico.
%             atrujo@uabc.mx
%
%  June 30, 2003.
%
%  To cite this file, this would be an appropriate format:
%  Trujillo-Ortiz, A. and R. Hernandez-Walls. (2003). GHtest: Games-Howell's approximate
%    test of equality of means from normal population when variances are heterogeneous. 
%    A MATLAB file. [WWW document]. URL http://www.mathworks.com/matlabcentral/fileexchange/
%    loadFile.do?objectId=3676&objectType=FILE
%
%  References:
%
%  Games, P. A. and Howell, J. F. (1976), Pairwise multiple comparision
%         procedures with unequal N's and/or variances: A Monte Carlo
%         study. Journal of Educational Statistics, 1:113-125.
%  Sokal, R. R. and Rohlf, F. J. (1981), Biometry. 2nd. ed. 
%         New-York:W. H. Freeman & Co. pp. 210-216.
%

if nargin < 3,
   alpha = 0.05; %(Default)
end;

if nargin < 2,
   d = 1; %(Data matrix by default)
end;

if nargin < 1, 
   error('Requires at least one argument.');
end;

if d == 1
   disp('You have choosed the data option.');
   disp(' ')
   [rx cx] = size(X);
   if cx ~= 2
      error('Data matrix X must to have 2 columns.');
   end;
   
   k = max(X(:,2));
   fprintf('The number of samples are:%2i\n\n', k);
   
   n = [];m = [];v = [];
   indice = X(:,2);
   for i = 1:k
      Xe = find(indice==i);
      eval(['X' num2str(i) '=X(Xe,1);']);
      eval(['n' num2str(i) '=length(X' num2str(i) ') ;'])
      eval(['m' num2str(i) '=mean(X' num2str(i) ');'])
      eval(['v' num2str(i) '=var(X' num2str(i) ');'])
      eval(['xn= n' num2str(i) ';'])
      eval(['xm= m' num2str(i) ';'])
      eval(['xv= v' num2str(i) ';'])
      n = [n;xn];m = [m;xm];v = [v;xv];
   end;
   
   df = [];D = [];q = [];CO = [];
   for i = k:-1:2
      for s = i-1:-1:1
         if s ~= i
            Co = [i s];
            CO = [CO;Co];
            eval(['df' num2str(i) num2str(s) '= (((v' num2str(i) '/n' num2str(i) ')+(v' num2str(s) '/n' num2str(s) '))^2)/(((v' num2str(i) '/n' num2str(i) ')^2/(n' num2str(i) '-1))+((v' num2str(s) '/n' num2str(s) ')^2)/(n' num2str(s) '-1));']);
            eval(['d' num2str(i) num2str(s) '= abs(m' num2str(i) '-m' num2str(s) ');']);
            eval(['q' num2str(i) num2str(s) '= (d' num2str(i) num2str(s) ')/sqrt((v' num2str(i) '/n' num2str(i) ')+(v' num2str(s) '/n' num2str(s) '));']);
            eval(['xdf= floor(df' num2str(i) num2str(s) ');']);
            eval(['xd= d' num2str(i) num2str(s) ';']);
            eval(['xq= q' num2str(i) num2str(s) ';']);
            df = [df;xdf];D = [D;xd];q = [q;xq];
         end;
      end;
   end;
   qe = [];S = [];
   for c = 1:k*((k-1)/2)
      x = qTukey(df(c),k,1-alpha);
      qe = [qe;x];
      if q(c) >= qe(c);
         s = 'S  ';
      else
         s = 'NS ';
      end;
      S = [S;s];
   end;
   
   disp('Table of the difference between each pair of means in terms of the Tukey''s');
   disp('statistic and then comparing them with its critical value.');
   fprintf('------------------------------------------------------------------------\n');
   disp(' Comparison            D               q              df''    Conclusion')
   fprintf('------------------------------------------------------------------------\n');
   for c = 1:(k*(k-1))/2
      fprintf('    %d  %d        %11.4f     %11.4f  %11i          %s\n',CO(c,1),CO(c,2),D(c),q(c),df(c),S(c,:))
   end;
   fprintf('------------------------------------------------------------------------\n');
   fprintf('With a given significance level of: %.4f\n', alpha);
   disp('The results are significant (S) and/or not significant (NS).');

else (d == 2);
   disp('You have choosed the statistics option.');
   disp(' ')
   [rx cx] = size(X);
   if cx ~= 4
      error('Statistics matrix X must to have 4 columns.');
   end;
   
   k = rx;
   fprintf('The number of samples are:%2i\n\n', k);
   
   n = [];m = [];v = [];
   indice = X(:,1);
   for i = 1:k
      Xe = find(indice==i);
      eval(['n' num2str(i) '=X(Xe,2);']);
      eval(['m' num2str(i) '=X(Xe,3);']);
      eval(['v' num2str(i) '=X(Xe,4);']);
      eval(['xn= n' num2str(i) ';']);
      eval(['xm= m' num2str(i) ';']);
      eval(['xv= v' num2str(i) ';']);
      n = [n;xn];m = [m;xm];v = [v;xv];
   end;
   
   df = [];D = [];q = [];CO = [];
   for i = k:-1:2
      for s = i-1:-1:1
         if s ~= i
            Co = [i s];
            CO = [CO;Co];
            eval(['df' num2str(i) num2str(s) '= (((v' num2str(i) '/n' num2str(i) ')+(v' num2str(s) '/n' num2str(s) '))^2)/(((v' num2str(i) '/n' num2str(i) ')^2/(n' num2str(i) '-1))+((v' num2str(s) '/n' num2str(s) ')^2)/(n' num2str(s) '-1));']);
            eval(['d' num2str(i) num2str(s) '= abs(m' num2str(i) '-m' num2str(s) ');']);
            eval(['q' num2str(i) num2str(s) '= (d' num2str(i) num2str(s) ')/sqrt((v' num2str(i) '/n' num2str(i) ')+(v' num2str(s) '/n' num2str(s) '));']);
            eval(['xdf= floor(df' num2str(i) num2str(s) ');']);
            eval(['xd= d' num2str(i) num2str(s) ';']);
            eval(['xq= q' num2str(i) num2str(s) ';']);
            df = [df;xdf];D = [D;xd];q = [q;xq];
         end;
      end;
   end;
   qe = [];S = [];
   for c = 1:k*((k-1)/2);
      x = qTukey(df(c),k,1-alpha);
      qe = [qe;x];
      if q(c) >= qe(c);
         s = 'S  ';
      else
         s = 'NS ';
      end;
      S = [S;s];
   end;
   
   disp('Table of the difference between each pair of means in terms of the Tukey''s');
   disp('statistic and then comparing them with its critical value.');
   fprintf('------------------------------------------------------------------------\n');
   disp(' Comparison            D               q              df''    Conclusion')
   fprintf('------------------------------------------------------------------------\n');
   for c = 1:(k*(k-1))/2
      fprintf('    %d  %d        %11.4f     %11.4f  %11i          %s\n',CO(c,1),CO(c,2),D(c),q(c),df(c),S(c,:))
   end;
   fprintf('------------------------------------------------------------------------\n');
   fprintf('With a given significance level of: %.4f\n', alpha);
   disp('The results are significant (S) and/or not significant (NS).');

end;