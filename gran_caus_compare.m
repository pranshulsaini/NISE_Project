% This file compares different granger causality codes I have. As I go for higher N, p2 makes more sense
clear all;


    
p1_rej = 0;
p2_rej = 0;
p3_rej = 0;
p4_rej = 0;

p1 =  ones(1000,1);
p2 =  ones(1000,1);
p3=  ones(1000,1);
p4=  ones(1000,1);


for i = 1:1000

    x = rand(1000,1);
    y = rand(1000,1);
    
    [F, c_v,p] = granger_cause_orig(x,y,0.05,5);
    if (p<=0.05)
        p1_rej = p1_rej +1;
    end
    p1(i) = p;
    
    
    [F, c_v,p] = granger_cause(x,y,0.05,5);
    if (p<=0.05)
        p2_rej = p2_rej +1;
    end
    p2(i) = p;
    
   
    [F, c_v,~,p] = granger_cause_1(x,y,0.05,5,1,5,1,1);
    if (p<=0.05)
         p3_rej = p3_rej +1;
    end
    p3(i) = p;
    
    [F, c_v,p] = granger_cause_no_intrcp(x,y,0.05,5);
    if (p<=0.05)
        p4_rej = p4_rej +1;
    end
    p4(i) = p;
       
end

plot(p1);
hold on;
plot(p2);
hold on;
plot(p3);
hold on;
plot(p4);
legend;
