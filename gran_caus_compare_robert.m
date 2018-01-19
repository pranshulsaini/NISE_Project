% This file compares different granger causality codes I have. 
% As I go for higher N, p2 makes more sense
clear all;



nLength =   500 ;
nTrials = 15000 ; 
p1 =  ones(nTrials,1);
p2 =  ones(nTrials,1);
p3 =  ones(nTrials,1);
p4 =  ones(nTrials,1);
p5 =  ones(nTrials,1);
p6 =  ones(nTrials,1);
p7 =  ones(nTrials,1);


for i = 1:nTrials

    x = rand(nLength,1);
    y = rand(nLength,1);
    
    [~,p] = granger_cause_orig(x,y,0.05,5);
    p1(i) = p;
    
    [~,p] = granger_cause(x,y,0.05,5);
    p2(i) = p;
      
    [~,p] = granger_cause_no_intrcp(x,y,0.05,5);
    p3(i) = p;
    
    %  search for best fit
    [F, c_v,p, pcor] = granger_cause_1(x,y,0.05,5,1,5,1,1);
    p4(i) = p;
    p5(i) = pcor;   
   
    % do NOT search for best fit, use the specified ( 5 )
    [F, c_v,p, pcor] = granger_cause_1(x,y,0.05,5,0,5,0,1);
    p6(i) = p;
    p7(i) = pcor;  
    
end




%
% plot distribution of p
%  should be  uniform [ 0 1  ]  for random data
%
%%
figure;
subplot(2,4,1)
hist(p1,50);
title('g.c.orig');

subplot(2,4,2)
hist(p2,50);
title('g.c');

subplot(2,4,3);
hist(p3,50);
title('g.c.no.intrcp');

subplot(2,4,4);
hist(p4,50);
title('g.c.1 use best');

subplot(2,4,5);
hist(p5,50);
title('g.c.1 use best; pcor');


subplot(2,4,6);
hist(p6,50);
title('g.c.1 use specified');

subplot(2,4,7);
hist(p7,50);
title('g.c.1 use specified; pcor');

%%
count = zeros(7,1)
count(1) = (100/15000)* sum(p1<0.05);
count(2) = (100/15000)*  sum(p2<0.05);
count(3) = (100/15000)*  sum(p3<0.05);
count(4) = (100/15000)* sum(p4<0.05);
count(5) = (100/15000)* sum(p5<0.05);
count(6) = (100/15000)* sum(p6<0.05);
count(7) = (100/15000)* sum(p7<0.05);

figure;
plot(count);

xlabel('The index of code/condition used ')
ylabel('Percentge of chance acceptance')

 















