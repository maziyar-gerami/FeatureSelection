function y=Mutate(x,mu, Data)
    
    feasible = true;
    
    while (feasible)
    
    nVar=numel(x);
    
    nmu=ceil(mu*nVar);
    
    j=randsample(nVar,nmu);
    
    y=x;
    y(j)=1-x(j);    
        
        if(Feasible(y,Data))
            
            feasible = false;
            
        end
    
    end
   
end